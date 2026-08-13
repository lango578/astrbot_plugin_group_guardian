# -*- coding: utf-8 -*-
"""本地 OCR（RapidOCR ONNX）支持：低配/离线场景下识别图片/视频帧中的广告文字。

- 使用 `rapidocr_onnxruntime`（ONNX 版，模型 ~30MB，常驻 ~300MB），
  适合 2 核 2G 低配服务器，不依赖智谱等云端视觉 API、不耗网络与额度；
- 引擎单例懒加载、全局复用（避免每次消息冷启动 ~1s）；同步调用放入线程池，不阻塞事件循环；
- 通过配置 `ocr_engine` 切换：`llm`（默认，云端视觉）/ `local`（本地 RapidOCR）/ `auto`（本地优先，本地无结果再回退云端）。
"""

import asyncio

from astrbot.api import logger

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover
    RapidOCR = None


class LocalOCRMixin:
    """本地 RapidOCR 识别能力，由 ``ModerationMixin`` 组合使用。"""

    def _init_local_ocr(self) -> None:
        """初始化本地 OCR：引擎槽位、并发锁、可用性标记。"""
        self._local_ocr_engine = None
        self._local_ocr_available = RapidOCR is not None
        self._local_ocr_lock = asyncio.Lock()

    def _local_ocr_configured(self, group_id: str = None) -> bool:
        """当前群是否启用本地 OCR（local/auto 引擎）。"""
        try:
            engine = self._cfg_str("ocr_engine", "llm", group_id=group_id)
        except Exception:
            engine = "llm"
        return str(engine or "").strip().lower() in ("local", "auto")

    async def _local_ocr_text(self, data: bytes) -> str:
        """对图片字节执行本地 OCR，返回识别文字；失败/不可用返回空串。"""
        if RapidOCR is None:
            logger.debug(
                "[GroupMgr] 本地 OCR 依赖 rapidocr_onnxruntime 未安装，"
                "请执行: pip install rapidocr_onnxruntime"
            )
            return ""
        if not data:
            return ""
        if self._local_ocr_engine is None:
            async with self._local_ocr_lock:
                if self._local_ocr_engine is None:
                    try:
                        self._local_ocr_engine = RapidOCR(
                            det_use_cuda=False,
                            rec_use_cuda=False,
                            use_cls=False,
                            show_log=False,
                        )
                    except Exception as exc:
                        logger.warning(f"[GroupMgr] 初始化 RapidOCR 失败: {exc}")
                        self._local_ocr_engine = False
        if not self._local_ocr_engine:
            return ""
        try:
            result, _ = await asyncio.to_thread(self._local_ocr_engine, data)
            if not result:
                return ""
            texts = []
            for line in result:
                if len(line) > 1 and line[1]:
                    texts.append(str(line[1]))
            return " ".join(texts).strip()
        except Exception as exc:
            logger.debug(f"[GroupMgr] 本地 OCR 识别失败: {exc}")
            return ""
