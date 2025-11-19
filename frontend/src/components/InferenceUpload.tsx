import { useRef, useState } from "react";
import { inferImage, InferenceResponse } from "../api";

interface InferenceUploadProps {
  onResult: (result: InferenceResponse) => void;
  onError: (message: string) => void;
}

export function InferenceUpload({ onResult, onError }: InferenceUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isLoading, setLoading] = useState(false);
  const [lastFileName, setLastFileName] = useState<string | null>(null);

  async function handleFile(file: File | null) {
    if (!file) return;
    setLastFileName(file.name);
    setLoading(true);
    try {
      const result = await inferImage(file);
      onResult(result);
    } catch (error) {
      console.error(error);
      onError("Inference failed. Ensure the backend is reachable and GPU dependencies are installed.");
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event: React.DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      handleFile(event.dataTransfer.files[0]);
      event.dataTransfer.clearData();
    }
  }

  return (
    <div className="card" style={{ position: 'relative', zIndex: 100 }}>
      <h2>Real-time Inference</h2>
      <p>Upload an image to classify with the TensorRT-optimized ResNet18 engine.</p>
      <label
        className="upload-zone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
        />
        <span>{lastFileName ?? "Drag & drop or click to select an image"}</span>
        <button
          className="button"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isLoading}
        >
          {isLoading ? "Running inference…" : "Select image"}
        </button>
      </label>
    </div>
  );
}
