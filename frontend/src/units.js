export function unitsFromGaps(text, gaps) {
  const chars = Array.from(text);
  const result = [];
  let start = 0;
  for (let i = 0; i < chars.length; i++)
    if (i === chars.length - 1 || gaps[i]) {
      result.push(chars.slice(start, i + 1).join(""));
      start = i + 1;
    }
  return result;
}
export function gapsFromUnits(units) {
  const gaps = Array(Math.max(0, Array.from(units.join("")).length - 1)).fill(
    false,
  );
  let pos = 0;
  units.slice(0, -1).forEach((unit) => {
    pos += Array.from(unit).length;
    gaps[pos - 1] = true;
  });
  return gaps;
}
export function commitRow(rows, index, changes) {
  return rows.map((row, i) => (i === index ? { ...row, ...changes } : row));
}
