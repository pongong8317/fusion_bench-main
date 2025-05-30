# -*- coding: utf-8 -*-
"""
WEMoE 指纹提取算法
继承 clip_weight_ensembling_moe，但:
  • 跳过 TTA / 评估
  • 统计每层 Router-Gate 的 Top-1 专家频次
  • 输出 (8, E, L) Tensor + CSV
"""
from __future__ import annotations
import csv, datetime as dt
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from fusion_bench.method.we_moe.clip_we_moe import (
    CLIPWeightEnsemblingMoEAlgorithm,
)
from fusion_bench.models.we_moe import WeightEnsemblingMoE  # Router 层类型
from fusion_bench.utils.instantiate import instantiate


class CLIPWeightEnsemblingMoEFingerprintAlgorithm(
    CLIPWeightEnsemblingMoEAlgorithm
):
    """
    参数要求:
        cfg.sample_per_task   : int  每任务样本数 (默认100)
        cfg.output_path       : str  目录
        cfg.output_formats    : List[str]  ["pt","csv"]
    """

    def run(self, modelpool, *args, **kwargs):  # pylint: disable=arguments-differ
        # 0. 读取 finger-print 相关配置
        fp_cfg = getattr(self.config, "fingerprint", {})
        n_per_task: int = int(fp_cfg.get("sample_per_task", 100))
        out_dir = Path(fp_cfg.get("output_path", "./"))
        formats: List[str] = fp_cfg.get("output_formats", ["pt", "csv"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 构建融合模型 (复用父类逻辑)
        moe_model = self.construct_moe_model(modelpool)
        moe_model.eval().requires_grad_(False)

        # 2. 获取 Router 层与专家数
        router_layers: List[WeightEnsemblingMoE] = [
            m for m in moe_model.modules() if isinstance(m, WeightEnsemblingMoE)
        ]
        num_layers = len(router_layers)
        if num_layers == 0:
            raise RuntimeError("未检测到 WeightEnsemblingMoE 路由层!")

        num_experts = router_layers[0].num_experts

        # 3. 初始化统计张量  shape = (任务, E, L)
        task_names: List[str] = list(modelpool.config.test_datasets)
        assert (
            len(task_names) == 8
        ), "需保证 8 个任务，当前为: {}".format(task_names)

        fingerprint = torch.zeros(
            (len(task_names), num_experts, num_layers), dtype=torch.int32
        )

        # 4. 注册 Hook
        hook_handles = []

        def make_hook(layer_idx: int, task_idx: int):
            # 返回一个闭包，每次前向时记录 batch 内所有样本
            def _hook(_module, _inputs, outputs):  # outputs shape: (B, *, E)
                logits = outputs
                while logits.ndim > 2:  # 若有 seq_len / token 维 -> 平均
                    logits = logits.mean(dim=1)
                top1 = logits.argmax(dim=-1)  # (B,)
                for idx in top1:
                    fingerprint[task_idx, idx.item(), layer_idx] += 1

            return _hook

        # 5. 遍历 8 个任务
        for task_idx, task_name in enumerate(task_names):
            # 5.1 装载固定前 n 条样本
            dataset = modelpool.load_test_dataset(task_name)
            if n_per_task > len(dataset):
                raise ValueError(
                    f"任务 {task_name} 测试集不足 {n_per_task} 条样本."
                )
            subset = Subset(dataset, list(range(n_per_task)))
            loader = DataLoader(
                subset, batch_size=1, shuffle=False, num_workers=0
            )
            # 5.2 为当前任务挂 hook
            for layer_idx, layer in enumerate(router_layers):
                h = layer.gate.register_forward_hook(
                    make_hook(layer_idx, task_idx)
                )
                hook_handles.append(h)

            # 5.3 前向推理
            for batch in loader:
                # taskpool 的 Dataset 返回 dict/image; 兼容两种
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.fabric.device) for k, v in batch.items()}
                    moe_model(**inputs)
                else:  # 仅 image tensor
                    moe_model(batch.to(self.fabric.device))

            # 5.4 清理 hook 供下一任务复用
            for h in hook_handles:
                h.remove()
            hook_handles.clear()

        # 6. 总计数校验
        expected_total = len(task_names) * n_per_task * num_layers
        total_count = int(fingerprint.sum().item())
        assert (
            total_count == expected_total
        ), f"计数不符! 得到 {total_count}, 期望 {expected_total}"

        # 7. 保存输出
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"fingerprint_{modelpool.config.base_model.replace('/','-')}_{ts}"
        if "pt" in formats:
            torch.save(fingerprint, out_dir / f"{base_name}.pt")
        if "csv" in formats:
            csv_path = out_dir / f"{base_name}.csv"
            with csv_path.open("w", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["task", "router_idx", "expert_idx", "count"])
                for t_idx, t_name in enumerate(task_names):
                    for l_idx in range(num_layers):
                        for e_idx in range(num_experts):
                            cnt = int(fingerprint[t_idx, e_idx, l_idx].item())
                            writer.writerow([t_name, l_idx, e_idx, cnt])

        # 8. 打印简洁校验信息
        print(
            f"[Fingerprint] tasks={len(task_names)}, experts={num_experts}, "
            f"layers={num_layers}, total_routes={total_count}"
        )

        # 返回元数据供上层打印
        return {
            "fingerprint_pt": str(out_dir / f"{base_name}.pt"),
            "fingerprint_csv": str(out_dir / f"{base_name}.csv"),
            "total_routes": total_count,
        }
