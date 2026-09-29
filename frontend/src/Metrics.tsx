export default function Metrics({
  metrics,
  stage,
}: {
  metrics: Record<string, any>;
  stage: string;
}) {
  const value = (n: unknown, percent = false) =>
    typeof n === "number"
      ? percent
        ? (n * 100).toFixed(1) + "%"
        : n.toLocaleString("zh-CN", { maximumFractionDigits: 4 })
      : "未评估";
  return (
    <div className="metrics-details">
      <div className="minor-metrics">
        <span>
          学习率 <b>{value(metrics.learning_rate)}</b>
        </span>
        <span>
          EPOCH <b>{value(metrics.epoch)}</b>
        </span>
      </div>
      <table>
        <thead>
          <tr>
            <th>指标</th>
            <th>训练集</th>
            <th>验证集</th>
          </tr>
        </thead>
        <tbody>
          {(stage === "pretrain"
            ? [["masked_accuracy", "掩码准确率"]]
            : [
                ["boundary_precision", "边界 Precision"],
                ["boundary_recall", "边界 Recall"],
                ["boundary_f1", "边界 F1"],
                ["segment_precision", "片段 Precision"],
                ["segment_recall", "片段 Recall"],
                ["segment_f1", "片段 F1"],
                ["sentence_accuracy", "整句完全正确率"],
                ["reconstruction_rate", "原文还原率"],
              ]
          ).map(([key, label]) => (
            <tr key={key}>
              <td>{label}</td>
              <td>{value(metrics["train_" + key] ?? metrics[key], true)}</td>
              <td>{value(metrics["val_" + key], true)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <small>
        GPU allocated / reserved：
        {metrics.gpu_allocated == null
          ? "不可用"
          : `${(metrics.gpu_allocated / 1048576).toFixed(0)} / ${(metrics.gpu_reserved / 1048576).toFixed(0)} MiB`}
        <br />
        设备显存总使用：
        {metrics.gpu_device_used == null
          ? "不可用"
          : `${(metrics.gpu_device_used / 1048576).toFixed(0)} MiB`}{" "}
        · GPU 利用率：
        {metrics.gpu_utilization == null
          ? "不可用"
          : metrics.gpu_utilization + "%"}
      </small>
    </div>
  );
}
