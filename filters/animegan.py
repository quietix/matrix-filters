import os
import cv2
import typing
import numpy as np
import onnxruntime as ort


class AnimeGAN:
    def __init__(
            self,
            model_path: str = '',
            downsize_ratio: float = 1.0,
    ) -> None:
        """
        Args:
            model_path: (str) - path to ONNX model file
            downsize_ratio: (float) - ratio to downsize input frame for faster inference
        """
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exist in {model_path}")

        self.downsize_ratio = downsize_ratio

        # Set execution provider based on device availability
        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']

        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        """Resize dimension to the nearest multiple of 32."""
        return 256 if x < 256 else x - x % 32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        """Resize the image to a size suitable for the model (32x32 multiple)."""
        h, w = frame.shape[:2]
        if x32:  # resize image to be a multiple of 32
            frame = cv2.resize(frame,
                               (self.to_32s(int(w * self.downsize_ratio)), self.to_32s(int(h * self.downsize_ratio))))
        frame = frame.astype(np.float32) / 127.5 - 1.0  # Normalize to range [-1, 1]
        return frame

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        """Post-process the model output to return to uint8 and resize to original dimensions."""
        frame = (frame.squeeze() + 1.) / 2 * 255  # Convert back to range [0, 255]
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))  # Resize to original dimensions
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to process and generate the animated output."""
        # Process the input frame (resize, normalize)
        image = self.process_frame(frame)

        # Run the model inference
        try:
            outputs = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(image, axis=0)})
        except Exception as e:
            print(f"Error during model inference: {e}")
            return frame  # Return the original frame in case of an error

        # Post-process the output frame (resize and convert back to uint8)
        frame = self.post_process(outputs[0], frame.shape[:2][::-1])

        return frame
