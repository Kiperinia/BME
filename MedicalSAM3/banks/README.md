train_bank 仅用于 strict retrieval protocol。

- train_bank/positive 与 train_bank/negative 只允许来自 Kvasir/CVC train split 的代表性 exemplar。
- train_bank 内禁止放入 val split、PolypGen 或任何 external test image。
- continual_bank 只用于后续 continual adaptation，不参与当前 strict external run。