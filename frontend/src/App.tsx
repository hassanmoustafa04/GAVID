import { useEffect, useMemo, useState } from "react";
import { fetchGpuMetrics, GPUResponse, GPUSample, InferenceResponse } from "./api";
import { InferenceUpload } from "./components/InferenceUpload";
import { GpuMetricsPanel } from "./components/GpuMetricsPanel";

interface ToastState {
  message: string;
  variant: "success" | "error";
}

export default function App() {
  const [inference, setInference] = useState<InferenceResponse | null>(null);
  const [history, setHistory] = useState<GPUSample[]>([]);
  const [source, setSource] = useState<string>("unknown");
  const [device, setDevice] = useState<string | undefined>(undefined);
  const [toast, setToast] = useState<ToastState | null>(null);

  useEffect(() => {
    if (!toast) {
      return;
    }
    const timeout = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const result = await fetchGpuMetrics();
        if (!active) {
          return;
        }
        handleMetrics(result);
      } catch (error) {
        console.error(error);
        if (active) {
          setToast({
            message: "Unable to retrieve GPU telemetry. Is DCGM or NVML available?",
            variant: "error"
          });
        }
      }
    }

    poll();
    const interval = window.setInterval(poll, 2000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleMetrics(result: GPUResponse) {
    setSource(result.source);
    setDevice(result.device);
    setHistory((prev) => {
      const data = [...prev, ...result.samples];
      return data.slice(-120); // keep 4 minutes of data
    });
  }

  function handleInference(result: InferenceResponse) {
    setInference(result);
    setToast({
      message: `Inference completed via ${result.engine} backend in ${result.latency_ms.toFixed(
        1
      )} ms`,
      variant: "success"
    });
  }

  const top5 = useMemo(() => inference?.top5 ?? [], [inference]);

  return (
    <div className="app">
      <header className="header">
        <h1>GPU-Accelerated Vision Inference Dashboard</h1>
        <p>
          Monitor GPU health and run real-time image classification accelerated with PyTorch,
          TensorRT, and FP16 execution. Backend metrics are sourced from NVIDIA DCGM when available.
        </p>
      </header>

      {toast ? (
        <div
          className="card"
          style={{
            borderColor: toast.variant === "success" ? "rgba(44, 182, 125, 0.4)" : "rgba(242,95,76,0.4)",
            background:
              toast.variant === "success"
                ? "rgba(22, 36, 32, 0.8)"
                : "rgba(40, 18, 24, 0.8)"
          }}
        >
          {toast.message}
        </div>
      ) : null}

      <div className="grid">
        <InferenceUpload
          onResult={(result) => {
            handleInference(result);
          }}
          onError={(message) => {
            setToast({ message, variant: "error" });
          }}
        />

        <div className="card">
          <h2>Inference Output</h2>
          {inference ? (
            <div className="inference-result">
              <div className="metric-chip">
                <span>Latency</span>
                <strong>{inference.latency_ms.toFixed(1)} ms</strong>
              </div>
              <div className="metric-chip">
                <span>Throughput</span>
                <strong>{inference.throughput_fps.toFixed(1)} FPS</strong>
              </div>
              <div className="metric-chip">
                <span>Engine</span>
                <strong>{inference.engine}</strong>
              </div>
              {top5.map((candidate, index) => (
                <div key={candidate.label} className="candidate">
                  <strong>
                    #{index + 1} {candidate.label}
                  </strong>
                  <small>{(candidate.confidence * 100).toFixed(2)}%</small>
                </div>
              ))}
            </div>
          ) : (
            <p>No inference yet. Upload an image to begin.</p>
          )}
        </div>
      </div>

      <GpuMetricsPanel samples={history} source={source} device={device} />
    </div>
  );
}
