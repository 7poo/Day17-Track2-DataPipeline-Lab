# Reflection — Day 17 (≤ 200 words)

Answer briefly, in your own words. This is graded on reasoning, not length.

1. **The flywheel.** Day 13 emitted agent traces; today you turned them into an
   eval set and DPO pairs that Day 22 will train on. Which step in
   `traces → Bronze → datasets` would break most silently in production if you
   got it wrong — and how would you detect it?

2. **Decontamination.** Your run dropped 2 of 3 preference pairs because their
   prompts were in the eval set. What concretely goes wrong if you *skip* this
   step and train on those pairs? How would the lie show up in your metrics?

3. **Point-in-time.** The naive join leaked a future `lifetime_spend` into the
   training row. Describe one feature in a system you know that would be
   dangerous to join without an `ASOF`/point-in-time guard.

4. **Graph vs vector.** From `kg_demo.py`, name one question the knowledge graph
   answers well that flat chunk retrieval (`embed.py`) would struggle with, and
   one where the graph is overkill.

_Write your answers below._

---

**1. The flywheel — bước nào nguy hiểm nhất?**

Bước nguy hiểm nhất là `flatten()` trong `traces.py` — đệ quy làm phẳng cây span. Nếu schema span thay đổi (ví dụ field `gen_ai.usage.input_tokens` đổi tên), `flatten()` trả về `None` thay vì báo lỗi, Bronze vẫn ghi được nhưng mọi downstream metric (cost, latency) đều sai. Để phát hiện: thêm check sau mỗi ingest rằng `input_tokens` không null cho ≥ 90% span có model; nếu tỷ lệ null đột tăng là dấu hiệu schema drift.

**2. Decontamination — chuyện gì xảy ra nếu bỏ qua?**

Nếu train trên các cặp DPO có prompt trùng eval set, mô hình học thuộc câu trả lời đúng cho đúng câu hỏi đó. Eval score sẽ inflate — trông rất tốt trên benchmark nhưng thực ra model chỉ đang nhớ training data. Dấu hiệu: train loss thấp bất thường trên eval prompts; khi đổi sang held-out prompts mới, hiệu suất rớt đáng kể.

**3. Point-in-time — ví dụ thực tế?**

Credit score của người dùng tại thời điểm xét duyệt vay. Nếu join bằng credit score mới nhất thay vì score tại ngày xét duyệt, mô hình sẽ học từ thông tin tương lai (score sau khi đã được/từ chối vay), dẫn đến mô hình credit risk ảo tưởng — tốt trên backtest nhưng thảm họa khi production.

**4. Graph vs vector — khi nào dùng gì?**

Graph thắng: "Widget ship từ đâu?" — cần 2-hop (widget → IS_A accessory → SHIPS_FROM Hanoi), không chunk nào chứa đủ cả hai thông tin. Vector RAG chỉ trả về một chunk nên không nối được. Graph là overkill: "Gadget có bảo hành bao lâu?" — câu trả lời nằm gọn trong một câu đơn, vector retrieval top-1 là đủ và nhanh hơn nhiều.
