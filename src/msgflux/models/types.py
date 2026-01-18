class ChatCompletionModel:
    model_type = "chat_completion"


class BatchedChatCompletionModel:
    model_type = "batched_chat_completion"


class ModerationModel:
    model_type = "moderation"


# Classifiers


class AudioClassifierModel:
    model_type = "audio_classifier"


class ImageClassifierModel:
    model_type = "image_classifier"


class VideoClassifierModel:
    model_type = "video_classifier"


class TabularClassifierModel:
    model_type = "tabular_classifier"


class TextClassifierModel:
    model_type = "text_classifier"


class ZeroShotImageClassifierModel:
    model_type = "zero_shot_image_classifier"


class ZeroShotTextClassifierModel:
    model_type = "zero_shot_text_classifier"


# Embedders


class AudioEmbedderModel:
    model_type = "audio_embedder"


class ImageEmbedderModel:
    model_type = "image_embedder"


class TextEmbedderModel:
    model_type = "text_embedder"


# Audio Gen


class TextToSpeechModel:
    model_type = "text_to_speech"


class AudioToAudioModel:
    model_type = "audio_to_audio"


class TextToMusicModel:
    model_type = "text_to_music"


class VideoTextToAudioModel:
    model_type = "video_text_to_audio"


# Image Gen


class TextToImageModel:
    model_type = "text_to_image"


class ImageTextToImageModel:
    model_type = "image_text_to_image"


class ImageToImageModel:
    model_type = "image_to_image"


# Text Gen


class AudioTextToTextModel:
    model_type = "audio_text_to_text"


class SpeechToTextModel:
    model_type = "speech_to_text"


class ImageToTextModel:
    model_type = "image_to_text"


class OCRModel:
    model_type = "ocr"


class TextTranslationModel:
    model_type = "text_translation"


class VideoTextToTextModel:
    model_type = "video_text_to_text"


# Video Gen


class ImageTextToVideoModel:  # VideoGenModel
    model_type = "image_text_to_video"


class TextToVideoModel:  # VideoGenModel
    model_type = "text_to_video"


class VideoTextToVideoModel:  # VideoGenModel
    model_type = "video_text_to_video"


# 3D


class ImageTo3DModel:
    model_type = "image_text_to_3d"


class ImageTextTo3DModel:
    model_type = "image_text_to_3d"


class TextTo3DModel:
    model_type = "text_to_3d"


# Others


class AnyToAnyModel:
    model_type = "any_to_any"


class DepthEstimationModel:
    model_type = "depth_estimation"


class ImageSegmenterModel:
    model_type = "image_segmenter"


class MaskGenModel:
    model_type = "mask_gen"


class ObjectDetectorModel:
    model_type = "object_detector"


class VADModel:
    model_type = "vad"


class TabularRegressorModel:
    model_type = "tabular_regressor"


class TextRerankerModel:
    model_type = "text_reranker"
