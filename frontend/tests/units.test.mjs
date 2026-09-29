import { test } from "node:test";
import assert from "node:assert/strict";
import { unitsFromGaps, gapsFromUnits, commitRow } from "../src/units.js";
test("码点跨度和原文无损", () => {
  const units = ["😀", " ", "e\u0301", "\n"];
  assert.deepEqual(unitsFromGaps(units.join(""), gapsFromUnits(units)), units);
});
test("两条金标切分与重复保留", () => {
  for (const units of [
    ["我", "养", "了", "一", "只", "猫", "叫", "魔丸"],
    [
      "我",
      "养",
      "了",
      "十二",
      "只",
      "猫",
      "，",
      "它",
      "们",
      "都",
      "很",
      "乖",
      "很",
      "可爱",
    ],
  ])
    assert.deepEqual(
      unitsFromGaps(units.join(""), gapsFromUnits(units)),
      units,
    );
});
test("空文本", () => assert.deepEqual(unitsFromGaps("", []), []));
test("切换样本前提交边界并保留身份和来源", () => {
  const rows = [
    { id: "1", text: "十二", units: ["十", "二"], source: "doc", split: "val" },
    { id: "2", text: "猫", units: ["猫"] },
  ];
  const saved = commitRow(rows, 0, {
    text: "十二",
    units: ["十二"],
    split: "val",
    status: "confirmed",
  });
  assert.deepEqual(saved[0], {
    id: "1",
    text: "十二",
    units: ["十二"],
    source: "doc",
    split: "val",
    status: "confirmed",
  });
  assert.equal(rows[0].units.length, 2);
});
