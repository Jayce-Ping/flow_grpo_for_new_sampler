from PIL import Image
import io
import json
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

# ===========================================================================
# GenEval reward — IN-ENV (no HTTP reward-server), ported from Flow-Factory's
# src/flow_factory/rewards/geneval.py: Mask2Former (mmdet 3.x) detection +
# open_clip ViT-L-14 zero-shot color, evaluating each image against its
# {tag, include:[{class,count,color?,position?}], exclude?} metadata.
# Returns the same 5-tuple the reward-server did:
#   (scores, rewards, strict_rewards, group_rewards, group_strict_rewards)
# where `scores` = the per-image STRICT reward (the GRPO training signal),
# `rewards` = the lenient fraction (accuracy logging), grouped by `tag`.
# ===========================================================================
GENEVAL_COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]
# Fixed canonical tag set + order (the 6 GenEval categories). Every rank's returned group dict uses
# these exact keys in this exact order so score_details keys are IDENTICAL across ranks -> the eval
# per-key accelerator.gather stays in lockstep. (A rank-varying key set desyncs the collectives.)
GENEVAL_TAGS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
GENEVAL_DETECTION_THRESHOLD = 0.3
GENEVAL_COUNTING_THRESHOLD = 0.9
GENEVAL_MAX_OBJECTS = 16
GENEVAL_DETECTOR_CONFIG = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco"
GENEVAL_DETECTOR_CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v3.0/mask2former/"
    "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/"
    "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
)
GENEVAL_CLIP_MODEL = "ViT-L-14"
import os as _os
from pathlib import Path as _Path
GENEVAL_OBJECT_NAMES_PATH = str(
    _Path(__file__).resolve().parents[1] / "dataset" / "geneval" / "object_names.txt"
)


def _geneval_check_position(bbox_a, bbox_b, relation):
    """True if bbox_b satisfies the spatial `relation` relative to bbox_a."""
    ca = ((bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2)
    cb = ((bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2)
    if relation == "above":
        return cb[1] < ca[1]
    if relation == "below":
        return cb[1] > ca[1]
    if relation == "left of":
        return cb[0] < ca[0]
    if relation == "right of":
        return cb[0] > ca[0]
    return False


def _geneval_to_pil_list(images):
    """Accept a torch NCHW float[0,1] tensor or a numpy array (NHWC/NCHW) -> list[PIL]."""
    if isinstance(images, torch.Tensor):
        arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    else:
        arr = np.asarray(images)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(0, 2, 3, 1)
        if arr.dtype != np.uint8:
            arr = (arr * 255).round().clip(0, 255).astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


class _GenEvalEngine:
    """In-process GenEval detector + CLIP color classifier (loaded once per device)."""

    def __init__(self, device):
        import torch.nn.functional as F  # noqa: F401 (used by methods)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        with open(GENEVAL_OBJECT_NAMES_PATH, "r") as f:
            self._object_names = [ln.strip() for ln in f if ln.strip()]
        self._name_to_idx = {n: i for i, n in enumerate(self._object_names)}
        self._det_thresh = GENEVAL_DETECTION_THRESHOLD
        self._count_thresh = GENEVAL_COUNTING_THRESHOLD
        self._max_objects = GENEVAL_MAX_OBJECTS
        self._init_detector()
        self._init_clip()
        self._color_text_features = {}

    # ---- model loading (mirrors Flow-Factory geneval.py) ----
    def _init_detector(self):
        try:
            from mmdet.apis import init_detector, inference_detector
        except ImportError as e:
            raise ImportError("mmdet is required for in-env GenEval: pip install mmdet mmengine") from e
        self._inference_detector = inference_detector
        cfg = GENEVAL_DETECTOR_CONFIG
        if not _os.path.exists(cfg) and not cfg.endswith(".py") and "/" not in cfg:
            resolved = self._resolve_mmdet_short_config(cfg)
            if resolved is None:
                raise FileNotFoundError(
                    f"Could not resolve mmdet config '{cfg}'. Ensure mmdet shipped its "
                    f"bundled model zoo (<mmdet>/.mim/configs/...)."
                )
            cfg = resolved
        self._detector = init_detector(cfg, GENEVAL_DETECTOR_CHECKPOINT, device=str(self.device))

    @staticmethod
    def _resolve_mmdet_short_config(short_name):
        import mmdet
        root0 = _Path(mmdet.__file__).parent
        for root in (root0 / ".mim" / "configs", root0.parent / "configs"):
            if root.is_dir():
                for p in root.rglob(f"{short_name}.py"):
                    return str(p)
        return None

    def _init_clip(self):
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("open_clip_torch is required for in-env GenEval color") from e
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            GENEVAL_CLIP_MODEL, pretrained="openai", device=str(self.device)
        )
        self._clip_tokenizer = open_clip.get_tokenizer(GENEVAL_CLIP_MODEL)
        self._clip_model.eval()

    # ---- scoring primitives ----
    @torch.no_grad()
    def _color_text(self, classname):
        import torch.nn.functional as F
        if classname not in self._color_text_features:
            prompts = [f"a photo of a {c} {classname}" for c in GENEVAL_COLORS]
            tok = self._clip_tokenizer(prompts).to(self._clip_model.visual.proj.device)
            feats = F.normalize(self._clip_model.encode_text(tok), dim=-1)
            self._color_text_features[classname] = feats
        return self._color_text_features[classname]

    @torch.no_grad()
    def _classify_color(self, image, bbox, classname):
        import torch.nn.functional as F
        x1, y1, x2, y2 = [int(c) for c in bbox]
        crop = image.crop((x1, y1, x2, y2))
        if crop.width < 1 or crop.height < 1:
            return "unknown"
        dev = self._clip_model.visual.proj.device
        img = self._clip_preprocess(crop).unsqueeze(0).to(dev)
        feats = F.normalize(self._clip_model.encode_image(img), dim=-1)
        sim = (feats @ self._color_text(classname).T).squeeze(0)
        return GENEVAL_COLORS[int(sim.argmax().item())]

    @torch.no_grad()
    def _detect(self, image):
        result = self._inference_detector(self._detector, np.array(image))
        pred = result.pred_instances
        bb = pred.bboxes.cpu().numpy()
        lb = pred.labels.cpu().numpy()
        sc = pred.scores.cpu().numpy()
        m = sc >= self._det_thresh
        return bb[m], lb[m], sc[m]

    @torch.no_grad()
    def evaluate_one(self, image, metadata, only_strict):
        """Return (lenient_fraction, strict_reward) for one image vs its metadata."""
        include = metadata.get("include", []) if isinstance(metadata, dict) else []
        exclude = metadata.get("exclude", None) if isinstance(metadata, dict) else None
        if isinstance(include, str):
            include = json.loads(include)
        if isinstance(exclude, str):
            exclude = json.loads(exclude) or None
        bboxes, labels, scores = self._detect(image)

        strict_flags, lenient_subs = [], []
        for req in include:
            classname = req["class"]
            expected = req.get("count", 1)
            color = req.get("color", None)
            position = req.get("position", None)
            cidx = self._name_to_idx.get(classname)
            if cidx is None:
                strict_flags.append(False); lenient_subs.append(0.0); continue
            cmask = labels == cidx
            cbb, csc = bboxes[cmask], scores[cmask]
            if expected > 1 or (exclude and any(e.get("class") == classname for e in exclude)):
                cbb = cbb[csc >= self._count_thresh]
            if len(cbb) > self._max_objects:
                cbb = cbb[: self._max_objects]
            found = len(cbb)
            count_reward = max(0.0, 1.0 - abs(expected - found) / expected)
            count_correct = (found == expected)

            # only_strict speedup: a wrong count can never be strict-correct -> skip CLIP/position.
            if only_strict and not count_correct:
                strict_flags.append(False); lenient_subs.append(0.0); continue

            if color and found > 0:
                colored = sum(
                    1 for bb in cbb[:expected]
                    if self._classify_color(image, bb, classname) == color
                )
                color_reward = max(0.0, 1.0 - abs(expected - colored) / expected)
                lenient_subs.append(min(count_reward, color_reward))
                strict_flags.append(count_correct and colored == expected)
            elif position and found > 0:
                relation, ref_idx = position
                ok = False
                if ref_idx < len(include):
                    ridx = self._name_to_idx.get(include[ref_idx]["class"])
                    if ridx is not None:
                        rbb = bboxes[labels == ridx]
                        if len(rbb) > 0 and len(cbb) > 0:
                            ok = _geneval_check_position(rbb[0], cbb[0], relation)
                lenient_subs.append(count_reward if ok else 0.0)
                strict_flags.append(count_correct and ok)
            else:
                lenient_subs.append(count_reward)
                strict_flags.append(count_correct)

        exclude_ok = True
        if exclude:
            for exc in exclude:
                cidx = self._name_to_idx.get(exc["class"])
                if cidx is None:
                    continue
                max_allowed = exc.get("count", 0)
                found = int((scores[labels == cidx] >= self._count_thresh).sum())
                if found > max_allowed:
                    exclude_ok = False
                    lenient_subs.append(max(0.0, 1.0 - (found - max_allowed) / max(max_allowed, 1)))

        strict = 1.0 if (strict_flags and all(strict_flags) and exclude_ok) else 0.0
        lenient = (sum(lenient_subs) / len(lenient_subs)) if lenient_subs else 0.0
        return lenient, strict


_GENEVAL_ENGINE = {}


def _get_geneval_engine(device):
    key = str(device)
    if key not in _GENEVAL_ENGINE:
        _GENEVAL_ENGINE[key] = _GenEvalEngine(device)
    return _GENEVAL_ENGINE[key]


def geneval_score(device):
    """IN-ENV GenEval reward (no HTTP server). Loads Mask2Former + CLIP once per device
    and evaluates each image against its metadata. Same 5-tuple contract as before."""
    engine = _get_geneval_engine(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        pil_images = _geneval_to_pil_list(images)
        all_scores, all_rewards, all_strict_rewards = [], [], []
        # Fixed-key ordered group dicts (all GENEVAL_TAGS present, empty ones kept) so every rank
        # iterates the SAME score_details keys in the SAME order -> collective gathers stay synced.
        group_rewards = {t: [] for t in GENEVAL_TAGS}
        group_strict_rewards = {t: [] for t in GENEVAL_TAGS}
        # mmdet's Mask2Former (ms_deform_attn) has no bf16 CUDA kernel and CLIP weights stay
        # fp32; disable AMP for the whole reward pass (matches Flow-Factory / GenEval reference).
        with torch.amp.autocast("cuda", enabled=False):
            for img, meta in zip(pil_images, metadatas):
                lenient, strict = engine.evaluate_one(img, meta, only_strict)
                reward = strict if only_strict else lenient
                tag = meta.get("tag", "overall") if isinstance(meta, dict) else "overall"
                all_scores.append(strict)          # GRPO training signal = strict reward
                all_rewards.append(reward)         # accuracy (lenient), or strict when only_strict
                all_strict_rewards.append(strict)
                if not only_strict and tag in group_rewards:
                    # Per-tag group rewards are produced ONLY in eval (fixed 6-tag key set/order).
                    group_rewards[tag].append(reward)
                    group_strict_rewards[tag].append(strict)
        if only_strict:
            # Training: EMPTY group dicts -> no per-tag score_details keys -> the training-phase
            # gather sees only equal-length per-image keys (no desync / no deadlock). Matches the
            # "only the strict reward is needed in training" intent.
            return all_scores, all_rewards, all_strict_rewards, {}, {}
        return all_scores, all_rewards, all_strict_rewards, group_rewards, group_strict_rewards

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
