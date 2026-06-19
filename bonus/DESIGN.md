# Bonus Design — Pipeline Flywheel cho Chatbot CSKH Tiếng Việt

## 1. Bài toán và ràng buộc thực

**Bối cảnh:** Một công ty thương mại điện tử Việt Nam vận hành chatbot CSKH
(chăm sóc khách hàng) xử lý ~50 000 lượt chat/ngày, chủ yếu bằng tiếng Việt
có dấu. Chatbot dùng mô hình LLM fine-tuned nội bộ, đang chạy nhưng tỷ lệ
"escalate to human" vẫn ở mức 35% — quá cao. Mục tiêu: dùng chính traffic sản
xuất để cải thiện mô hình liên tục mà **không cần annotation viên làm việc thủ
công 24/7**.

**Ràng buộc lộn xộn:**
- Tiếng Việt có dấu: tokenizer cần Unicode-aware; "return" vs "trả hàng" không
  matching bằng ASCII n-gram thông thường.
- PDPL (Luật 91/2025 về bảo vệ dữ liệu cá nhân): không được lưu tên, số điện
  thoại, địa chỉ khách hàng vào tập training mà không có consent. PII phải được
  mask trước khi vào Bronze.
- Bandwidth thực tế: server đặt tại Hà Nội, nhiều khách ở vùng nông thôn dùng
  3G — latency inference phải < 2 giây, pipeline training không được ảnh hưởng
  serving.
- Budget: công ty startup, không có Snowflake hay Databricks; phải chạy trên
  DuckDB + Postgres RDS + một GPU A10 thuê theo giờ.

---

## 2. Sơ đồ kiến trúc

```
[ Chatbot serving ]
       │  emit trace/turn (gen_ai.* OTel)
       ▼
[ Kafka topic: raw-turns ]
       │  consumer (append-only)
       ▼
[ Bronze: raw spans + PII mask ]
       │
  ┌────┴────────────────────────┐
  ▼                             ▼
[ Silver: curate eval set ]  [ Silver: DPO pairs ]
  (split=eval, ok turns)       (ok vs error, same prompt)
       │                             │
       ▼                             ▼
[ Decontaminate ]  ◄──────────────────┘
       │  drop pairs whose prompt ∈ eval
       ▼
[ datasets/eval_golden.jsonl ]   [ datasets/preference_pairs.jsonl ]
       │                                     │
       └──────────────┬──────────────────────┘
                      ▼
             [ Fine-tune job (Day 22 DPO) ]
                      │
                      ▼
             [ New model checkpoint ]
                      │
                      ▼
             [ A/B test vs current model ]
                      │  if win → promote
                      ▼
             [ Chatbot serving ] ── (vòng lặp)
```

---

## 3. Các câu hỏi then chốt và quyết định

### Q1. Batch hay streaming? Độ tươi cần bao nhiêu?

**Quyết định: micro-batch 15 phút, không phải real-time streaming.**

*Lý do:* Fine-tuning DPO tốn ~4 giờ trên A10; độ tươi dưới 15 phút của dữ
liệu training không mang lại lợi ích gì. Kafka vẫn dùng cho ingest (để đảm
bảo at-least-once và replay), nhưng consumer chạy theo cửa sổ 15 phút thay vì
per-event. Lambda/Kappa architecture là overkill — chi phí vận hành hai stack
(batch + stream) gấp đôi mà gain gần bằng 0 với bài toán này.

*Đánh đổi X vs Y:* Streaming mỗi event (Kappa) cho latency ~1 phút nhưng
phức tạp hơn và không cải thiện model vì training không thể chạy liên tục.
Micro-batch 15 phút (Lambda lite) đơn giản hơn, đủ tươi cho daily training
cycle.

### Q2. Validate gì trước khi vào model? Dòng xấu đi đâu?

**Quyết định: quality gate 3 tầng + DLQ topic trên Kafka.**

Tầng 1 — **PII mask** (bắt buộc vì PDPL): regex + NER detect số điện thoại
(`0[3-9]\d{8}`), CCCD, họ tên (NER tiếng Việt). Mask trước khi ghi Bronze.
Không mask → vi phạm luật → phạt tối đa 5% doanh thu.

Tầng 2 — **Schema contract** (Pandera): turn phải có user_input, agent_output,
trace_id, status. Thiếu field → quarantine.

Tầng 3 — **Quality score** (rule-based): turns quá ngắn (< 5 token) hoặc là
copypaste lỗi (agent_output = user_input) → quarantine với lý do để analyst
review.

DLQ = Kafka topic `bad-turns` (không phải chỉ là file CSV), để ops team có
thể replay khi fix được extractor. Một dòng xấu không bao giờ dừng pipeline.

### Q3. Train/serve parity — feature nào nguy hiểm?

**Quyết định: ASOF join bắt buộc cho mọi feature có timestamp.**

Ví dụ nguy hiểm nhất: `user_complaint_rate_7d` (tỷ lệ khiếu nại 7 ngày gần
nhất). Nếu join bằng giá trị hiện tại thay vì giá trị tại thời điểm xảy ra
turn, mô hình sẽ "biết trước" user sắp khiếu nại — offline AUC trông rất đẹp
nhưng production thảm họa. Fix: DuckDB ASOF JOIN với `valid_from` timestamp,
chính xác như `features.py` trong lab này.

### Q4. Decontamination — fuzzy hay exact?

**Quyết định: 13-gram decontamination cho tiếng Việt, ngưỡng n=8 (không phải 13).**

Tiếng Việt có câu ngắn hơn tiếng Anh (trung bình 12–15 token/câu vs 20–25).
n=13 dùng cho tiếng Anh chuẩn (WMT convention) nhưng với tiếng Việt, n=8 là
đủ để bắt paraphrase mà không quá aggressive (loại nhầm các turns khác nhau
thật sự). Exact-match bị miss khi khách viết "trả hàng" vs "hoàn hàng" — cùng
nghĩa nhưng không match.

*Phương án bị loại:* Embedding similarity (cosine > 0.9). Lý do loại: cần
embedding model chạy trên toàn bộ eval set mỗi lần có pair mới — O(n×m) cost,
chậm và tốn GPU mà gain không đáng. 13-gram chạy trên CPU, O(n+m), đủ tốt.

### Q5. RAG hay Knowledge Graph cho CSKH?

**Quyết định: hybrid — vector RAG cho câu hỏi đơn, KG cho policy multi-hop.**

"Gadget bảo hành bao nhiêu ngày?" → vector RAG, top-1 chunk đủ.

"Widget mua ở Đà Nẵng thì trả hàng về kho nào?" → cần 2-hop: widget → IS_A
accessory → SHIPS_FROM kho Hà Nội. Vector RAG trả về chunk "widget trả trong
30 ngày" nhưng không biết kho. KG traversal trả về đúng kho vì graph mã hoá
connection.

*Đánh đổi:* KG cần pipeline extract triples (LLM hoặc rule-based) và refresh
khi policy thay đổi. Vector RAG chỉ cần re-embed doc mới. Quyết định: dùng
cả hai — vector cho lookup nhanh, KG cho câu hỏi escalate có nhiều hop.

### Q6. Failure semantics — backfill an toàn thế nào?

**Quyết định: partition-aware DELETE+INSERT trên Bronze, idempotent per date.**

Nếu pipeline lỗi lúc 3h sáng và cần re-run cho ngày 2026-06-15: chạy lại
với `--date 2026-06-15` sẽ DELETE rows cho ngày đó rồi INSERT lại từ nguồn —
không nhân đôi, không mất data ngày khác. Đây chính là extension exercise 4
đã implement trong `extract.py`. Side-effect không thể đảo ngược duy nhất là
ghi file `quarantine.csv` — nhưng nó được ghi đè, không append, nên idempotent.

---

## 4. Phương án bị loại

**Apache Spark + Delta Lake** — bị loại vì: cluster cost > $800/tháng cho
50k turns/ngày là overkill hoàn toàn. DuckDB xử lý 50k rows/ngày trong
<1 giây trên laptop. Spark hợp lý từ 50M rows/ngày trở lên hoặc khi cần
distributed shuffle. Đây là cái bẫy "dùng big data tools cho small data" —
thêm complexity mà không thêm value.

**Full real-time streaming với Flink** — bị loại vì latency của fine-tuning
(4h/cycle) không benefit gì từ ingest latency <1 giây. Kafka micro-batch
đã đủ và đơn giản hơn nhiều.

---

## 5. Bối cảnh Việt Nam

- **Tiếng Việt có dấu**: n-gram phải Unicode-aware; `unicodedata.normalize('NFC', text)` trước khi tokenize để tránh split dấu ra khỏi ký tự.
- **PDPL Luật 91/2025**: mask PII bắt buộc trước Bronze. Không có "xử lý sau" — dữ liệu raw không được ghi có PII.
- **Hạ tầng thực tế**: latency VN-US ~250ms → không dùng OpenAI API cho inference production; phải self-host model. Pipeline training chạy off-peak (23h–5h) để không tranh GPU với serving.
- **Đặc thù CSKH VN**: khách hay viết tắt ("hg" = hàng, "k" = không, "đc" = được) — normalizer cần mapping viết tắt trước khi quality gate, không thì schema check "input quá ngắn" sẽ false-positive.
