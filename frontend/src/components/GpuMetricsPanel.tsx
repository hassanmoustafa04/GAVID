import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  CartesianGrid
} from "recharts";
import { GPUSample } from "../api";

interface GpuMetricsPanelProps {
  samples: GPUSample[];
  source: string;
  device?: string;
}

function formatTimestamp(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString();
}

export function GpuMetricsPanel({ samples, source, device }: GpuMetricsPanelProps) {
  const latest = samples.at(-1);
  const chartData = samples.map((sample) => ({
    time: formatTimestamp(sample.timestamp),
    gpu: sample.gpu_util,
    mem: sample.mem_util,
    power: sample.power_w,
    temp: sample.temperature_c
  }));

  return (
    <div className="card" style={{ position: 'relative', zIndex: 100 }}>
      <h2>GPU Telemetry</h2>
      <p>
        Source: <strong>{source}</strong> {device ? `· ${device}` : null}
      </p>

      {latest ? (
        <div className="metrics-grid">
          <div className="metric-chip">
            <span>GPU Util</span>
            <strong>{latest.gpu_util.toFixed(1)}%</strong>
          </div>
          <div className="metric-chip">
            <span>Memory Util</span>
            <strong>{latest.mem_util.toFixed(1)}%</strong>
          </div>
          <div className="metric-chip">
            <span>Memory Used</span>
            <strong>
              {latest.memory_used_mb.toFixed(0)} / {latest.memory_total_mb.toFixed(0)} MiB
            </strong>
          </div>
          <div className="metric-chip">
            <span>Power</span>
            <strong>
              {latest.power_w.toFixed(1)} / {latest.power_limit_w.toFixed(1)} W
            </strong>
          </div>
          <div className="metric-chip">
            <span>Temperature</span>
            <strong>{latest.temperature_c.toFixed(1)}°C</strong>
          </div>
        </div>
      ) : (
        <p>Waiting for telemetry samples…</p>
      )}

      <div className="metrics-chart">
        <ResponsiveContainer>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="time" stroke="rgba(255,255,255,0.6)" />
            <YAxis stroke="rgba(255,255,255,0.6)" />
            <Tooltip
              contentStyle={{ backgroundColor: "#121828", border: "none", borderRadius: 8 }}
            />
            <Legend />
            <Line type="monotone" dot={false} dataKey="gpu" stroke="#7f5af0" name="GPU %" />
            <Line type="monotone" dot={false} dataKey="mem" stroke="#2cb67d" name="Memory %" />
            <Line type="monotone" dot={false} dataKey="power" stroke="#ff8906" name="Power W" />
            <Line type="monotone" dot={false} dataKey="temp" stroke="#f25f4c" name="Temp °C" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
