#-------------upgraded code load perist, context + llm wrapper ----------------



import os, shutil, asyncio, re, threading
from llama_cpp import Llama


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"⚠️ Invalid integer for {name}={raw!r}; using {default}.")
        return default


def _env_bool(name, default):
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def _default_llama_runtime():
    is_apple_silicon = os.uname().sysname == "Darwin" and os.uname().machine == "arm64"
    cpu_threads = os.cpu_count() or 4

    if is_apple_silicon:
        return {
            "n_threads": min(cpu_threads, 8),
            "n_batch": 256,
            "n_gpu_layers": -1,
            "main_gpu": 0,
            "offload_kqv": True,
            "flash_attn": True,
        }

    return {
        "n_threads": cpu_threads,
        "n_batch": 256,
        "n_gpu_layers": 0,
        "main_gpu": 0,
        "offload_kqv": True,
        "flash_attn": False,
    }

class ModelSessionManager:
    """
    Unified model manager for TinyLlama/Phi-3:
    - Handles safe load/save of GGUF sessions.
    - Provides async LLM wrapper with history/context support.
    """

    def __init__(self, model_path, n_ctx=1024, session_file="Tinyllama_context.bin", verbose=False):
        self.model_path = model_path
        self.session_file = session_file
        self.verbose = verbose
        self._lock = asyncio.Lock()
        runtime_defaults = _default_llama_runtime()
        self.n_threads = _env_int("LLM_N_THREADS", runtime_defaults["n_threads"])
        self.n_batch = _env_int("LLM_N_BATCH", runtime_defaults["n_batch"])
        self.n_gpu_layers = _env_int("LLM_N_GPU_LAYERS", runtime_defaults["n_gpu_layers"])
        self.main_gpu = _env_int("LLM_MAIN_GPU", runtime_defaults["main_gpu"])
        self.offload_kqv = _env_bool("LLM_OFFLOAD_KQV", runtime_defaults["offload_kqv"])
        self.flash_attn = _env_bool("LLM_FLASH_ATTN", runtime_defaults["flash_attn"])

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            seed=42,
            verbose=self.verbose,
            chat_format=None,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
            offload_kqv=self.offload_kqv,
            flash_attn=self.flash_attn,
        )

        gpu_mode = "CPU only" if self.n_gpu_layers == 0 else (
            "GPU all layers" if self.n_gpu_layers < 0 else f"GPU offload {self.n_gpu_layers} layers"
        )
        print(
            "✅ Model loaded: "
            f"{os.path.basename(model_path)} "
            f"(threads={self.n_threads}, n_batch={self.n_batch}, n_ctx={n_ctx}, "
            f"{gpu_mode}, main_gpu={self.main_gpu}, offload_kqv={self.offload_kqv}, flash_attn={self.flash_attn})"
        )
        self._safe_load_context()

    # ---------- Context management ----------
    def _safe_load_context(self):
        if not hasattr(self.llm, "load_session"):
            print("ℹ️ Context caching not supported by this build.")
            return
        if not os.path.exists(self.session_file):
            print("ℹ️ No previous context found — starting fresh.")
            return
        try:
            with open(self.session_file, "rb") as f:
                data = f.read()
                if len(data) < 512:
                    raise ValueError("Context file too small (corrupt).")
            self.llm.load_session(data)
            print(f"✅ Loaded cached context from {self.session_file}")
        except Exception as e:
            print(f"⚠️ Failed to load cached context ({e}), starting clean.")
            try:
                os.remove(self.session_file)
            except OSError:
                pass

    def save_context(self):
        if not hasattr(self.llm, "save_session"):
            if self.verbose:
                print("ℹ️ save_session() not supported; skipping.")
            return
        try:
            data = self.llm.save_session()
            tmp_file = self.session_file + ".tmp"
            with open(tmp_file, "wb") as f:
                f.write(data)
            shutil.move(tmp_file, self.session_file)
            if self.verbose:
                print(f"💾 Context saved → {self.session_file}")
        except Exception as e:
            print(f"⚠️ Context save failed safely: {e}")

    async def save_context_async(self):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self.save_context)

    def get_model(self):
        return self.llm

    def _build_prompt(self, system_preamble: str, user_query: str,
                      context_text: str = "", history_text: str = "",
                      web_results_text: str = "") -> str:
        prompt_parts = []
        if system_preamble.strip():
            prompt_parts.append(f"System: {system_preamble.strip()}")
        if context_text.strip():
            prompt_parts.append(f"\nProduct Information:\n{context_text.strip()}")
        if history_text.strip():
            prompt_parts.append(f"\nRecent Conversation:\n{history_text.strip()}")
        if web_results_text.strip():
            prompt_parts.append(f"\nWeb Search Results:\n{web_results_text.strip()}")
            
        prompt_parts.append(f"\nUser: {user_query.strip()}")
        prompt_parts.append("\nAssistant:")
        return "".join(prompt_parts)

    async def _generate(self, prompt: str, temperature: float, max_tokens: int,
                        top_p: float, stop_tokens):
        loop = asyncio.get_running_loop()

        def _infer():
            return self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_tokens,
            )

        try:
            async with self._lock:
                out = await loop.run_in_executor(None, _infer)
            text = out.get("choices", [{}])[0].get("text", "") or ""
            cleaned = text.strip()
            cleaned = re.sub(r"^(Support Agent:|Agent:|Assistant:|support agent:|Aria:|aria:)", "", cleaned, flags=re.I).strip()
            cleaned = re.sub(r"\n+", " ", cleaned)
            return cleaned
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return ""

    # ---------- LLM wrapper ----------
    async def generate_reply(self, system_preamble: str, user_query: str,
                             context_text: str = "", history_text: str = "",
                             web_results_text: str = "",
                             temperature: float = 0.25, max_tokens: int = 100):
        """
        Unified async inference wrapper with context + history injection.
        Formats prompt cleanly to prevent model confusion and hallucination.
        """
        prompt = self._build_prompt(
            system_preamble,
            user_query,
            context_text=context_text,
            history_text=history_text,
            web_results_text=web_results_text,
        )
        return await self._generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.85,
            stop_tokens=["User:", "Assistant:", "System:"],
        )

    async def generate_fast_reply(self, system_preamble: str, user_query: str,
                                  temperature: float = 0.15, max_tokens: int = 64):
        prompt = self._build_prompt(system_preamble, user_query)
        return await self._generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
            stop_tokens=["User:", "Assistant:", "System:"],
        )

    async def stream_reply(self, system_preamble: str, user_query: str,
                           context_text: str = "", history_text: str = "",
                           web_results_text: str = "",
                           temperature: float = 0.35, max_tokens: int = 100,
                           top_p: float = 0.9, stop_tokens=None):
        prompt = self._build_prompt(
            system_preamble,
            user_query,
            context_text=context_text,
            history_text=history_text,
            web_results_text=web_results_text,
        )
        stop_tokens = stop_tokens or ["User:", "Assistant:", "System:"]

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        sentinel = object()

        def _stream_worker():
            try:
                for chunk in self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_tokens,
                    stream=True,
                ):
                    text = chunk.get("choices", [{}])[0].get("text", "") or ""
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        async with self._lock:
            worker = threading.Thread(target=_stream_worker, daemon=True)
            worker.start()

            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    print(f"[LLM ERROR] {item}")
                    break
                yield item
