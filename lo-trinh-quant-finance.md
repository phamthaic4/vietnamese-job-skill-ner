# Lộ trình Quant Research — Volatility Regime & Direction Prediction

**Mục tiêu:** Xây nền tảng quant research thật (không phải "ML áp vào tài chính"), chuẩn bị cho AI Quant Challenge, WorldQuant BRAIN, và các cơ hội quant trong tương lai.

**Thời lượng tổng:** ~8-10 tuần, part-time (10-15h/tuần). Có thể rút ngắn nếu full-time.

**Nguyên tắc xuyên suốt:** 70% thời gian dành cho methodology (leakage, baseline, backtest, giải thích), 30% cho việc train model. Nếu tỷ lệ này bị đảo ngược, project mất giá trị "quant" dù demo vẫn chạy đẹp.

---

## Tools & Setup

| Loại | Công cụ | Ghi chú |
|---|---|---|
| Data | `vnstock` (VN), `yfinance` (US/global) | vnstock cho data VN30, dễ lấy OHLCV |
| Xử lý | `pandas`, `numpy` | Chuẩn |
| Volatility model | `arch` (Python package cho GARCH) | `pip install arch` |
| ML baseline/model | `scikit-learn` (logistic regression), `lightgbm` | LightGBM là chuẩn công nghiệp cho tabular financial data |
| DL (tuỳ chọn, làm sau) | `pytorch` | Chỉ dùng sau khi có baseline vững |
| Backtest | Tự viết walk-forward split (khuyến nghị hiểu cơ chế trước khi dùng lib), hoặc `vectorbt` khi đã hiểu rõ | Không dùng lib ngay từ đầu — tự code walk-forward để hiểu bản chất |
| Visualization | `matplotlib`, `plotly` | Cho equity curve, drawdown chart |
| Tracking thí nghiệm | CSV log đơn giản hoặc `mlflow` | Không cần phức tạp |
| Luyện alpha research | WorldQuant BRAIN (platform.worldquantbrain.com) | Miễn phí, viết alpha expression, đo IC — mô phỏng đúng dạng thi quant challenge |
| Version control | GitHub | Bắt buộc, đây là sản phẩm cuối cùng để show |

---

## Nguồn học (theo thứ tự ưu tiên)

1. **"Advances in Financial Machine Learning" — Marcos López de Prado**
   Sách chuẩn của giới quant research. Đọc chương về: triple-barrier labeling, purged k-fold CV, sample weights, meta-labeling. Không cần đọc hết, chỉ cần các chương liên quan đến labeling & cross-validation.
2. **Documentation của `arch` package** — hiểu GARCH(1,1), cách fit, cách diễn giải tham số.
3. **Investopedia / QuantStart (bài viết)** — khái niệm: Sharpe ratio, max drawdown, look-ahead bias, survivorship bias, walk-forward analysis. Đọc nhanh, không cần sách riêng.
4. **WorldQuant BRAIN — phần "Learn"** — có tutorial riêng về cách viết alpha, đo IC, turnover. Làm song song với project chính.
5. **Kaggle: "Optiver Realized Volatility Prediction" (competition cũ)** — đọc top solutions/discussion để thấy dân thực chiến xử lý volatility prediction thế nào (không copy code, chỉ đọc ý tưởng).

---

## Lộ trình theo tuần

### Tuần 1 — Methodology nền tảng (Layer 1)
**Mục tiêu:** Hiểu đúng các bẫy trước khi chạm vào model.

Công việc:
- Đọc về look-ahead bias, survivorship bias, walk-forward validation, Sharpe/Sortino, max drawdown (nguồn #3)
- Lấy data OHLCV của 1 mã (ví dụ VNM hoặc FPT trên vnstock, hoặc SPY trên yfinance)
- Tự code 1 chiến lược cực đơn giản: moving average crossover (MA20/MA50)
- Backtest chiến lược đó bằng walk-forward split tự viết tay (không dùng lib) — mục đích là hiểu cơ chế, không phải để chiến lược này "ăn tiền"
- Tính Sharpe ratio, max drawdown, so sánh với buy-and-hold

Deliverable tuần: script backtest MA-crossover walk-forward + báo cáo ngắn (kết quả, có beat buy-and-hold không, tại sao)

---

### Tuần 2 — Data pipeline & target labeling
**Mục tiêu:** Chuẩn bị data đúng cách cho cả 2 bài toán (volatility regime + direction).

Công việc:
- Lấy data nhiều mã hơn (5-10 mã VN30 hoặc S&P 500 subset) để có đủ sample
- Tính realized volatility rolling window (20 ngày) — đây là input cho volatility regime
- Gán nhãn volatility regime (low/medium/high) dựa trên percentile của rolling vol — **chỉ dùng data quá khứ tại mỗi thời điểm để chia percentile, không dùng percentile tính trên toàn bộ dataset** (đây là điểm leakage rất dễ mắc)
- Tính target direction: excess return so với benchmark (index), không phải raw return
- Kiểm tra kỹ shift: feature tại thời điểm t chỉ dùng data ≤ t, target là t+1 trở đi

Deliverable tuần: dataset đã labeled sạch, script kiểm tra không có leakage (ví dụ correlation giữa feature và target tương lai = 0 nếu shuffle target)

---

### Tuần 3 — Baseline bắt buộc trước ML
**Mục tiêu:** Có baseline đúng chuẩn trước khi chạm ML — đây là bước phân biệt bạn với người chỉ biết train model.

Công việc:
- Fit GARCH(1,1) bằng `arch` package cho volatility → đây là baseline cho bài toán volatility regime
- Fit logistic regression đơn giản (feature: lag return, RSI) cho bài toán direction → baseline
- Đánh giá cả 2 baseline bằng metric đúng: cho volatility dùng RMSE hoặc so sánh regime classification accuracy; cho direction dùng cả accuracy và IC (Information Coefficient)
- Ghi lại số liệu baseline — đây là mốc để so sánh mọi thứ sau này

Deliverable tuần: bảng kết quả baseline (GARCH và logistic regression), lưu lại để so sánh

---

### Tuần 4-5 — Feature engineering & alpha research (Layer 2)
**Mục tiêu:** Xây feature có ý nghĩa kinh tế, không chỉ nhét số vào model.

Công việc:
- Tạo 8-12 feature: lag return (1,5,20 ngày), realized volatility rolling, volume z-score, RSI, MACD, khoảng cách giá so với MA
- Với mỗi feature, tự hỏi: "tại sao feature này có thể dự đoán được target?" — viết ra 1-2 câu lý giải kinh tế cho từng feature (quant research luôn quan tâm intuition, không chỉ số liệu)
- Đo Information Coefficient (IC) của từng feature riêng lẻ với target (correlation giữa feature và forward return) — đây chính là cách đánh giá alpha thực tế trong ngành
- Song song: bắt đầu làm quen WorldQuant BRAIN — thử viết vài alpha expression đơn giản trên platform, xem cách họ đo IC, turnover, để so sánh cách làm của bạn với chuẩn công nghiệp

Deliverable tuần: bảng IC của từng feature, ghi chú lý giải kinh tế, vài alpha thử nghiệm trên BRAIN

---

### Tuần 6 — Train model chính (Layer 3, phần 1)
**Mục tiêu:** Áp ML đúng chỗ, so sánh với baseline.

Công việc:
- Train LightGBM cho cả 2 bài toán (volatility regime classification, direction/excess return)
- Dùng walk-forward validation (không random split, không k-fold ngẫu nhiên — phải giữ tính thời gian)
- So sánh kết quả với baseline (GARCH, logistic regression) từ tuần 3 — bằng đúng metric
- Xem feature importance — có khớp với lý giải kinh tế ở tuần 4-5 không?

Deliverable tuần: bảng so sánh LightGBM vs baseline, feature importance chart

---

### Tuần 7 — Thử DL (tuỳ chọn, chỉ nếu còn thời gian)
**Mục tiêu:** Kiểm chứng DL có thực sự cần thiết không — đây cũng là 1 finding quan trọng.

Công việc:
- Thử LSTM đơn giản trên sequence feature
- So sánh với LightGBM — nếu LSTM không tốt hơn, đó là kết quả hợp lệ và đáng ghi lại (không phải thất bại)
- Không cần tối ưu sâu phần này, mục đích là có thêm 1 điểm so sánh trong báo cáo

Deliverable tuần: kết quả so sánh LSTM vs LightGBM vs baseline

---

### Tuần 8 — Backtest chiến lược đầy đủ & portfolio construction
**Mục tiêu:** Kết nối signal → performance thật, không dừng ở accuracy.

Công việc:
- Dùng signal từ model tốt nhất (LightGBM hoặc DL nếu tốt hơn) để xây 1 chiến lược đơn giản: long/short hoặc long-only dựa trên signal
- Backtest walk-forward đầy đủ, tính transaction cost (giả định 0.1-0.2% mỗi giao dịch)
- Tính Sharpe ratio, max drawdown, so với buy-and-hold
- Vẽ equity curve

Deliverable tuần: script backtest hoàn chỉnh, equity curve, bảng metric hiệu suất

---

### Tuần 9-10 — Viết báo cáo & polish GitHub repo
**Mục tiêu:** Đóng gói thành sản phẩm trình bày được.

Công việc:
- README rõ ràng: mục tiêu, phương pháp, baseline, kết quả, **giới hạn của project** (phần này quan trọng — thể hiện bạn hiểu rõ mình đang làm gì)
- Tách code thành module rõ ràng: `data.py`, `features.py`, `baseline.py`, `model.py`, `backtest.py`
- Notebook hoặc report tổng hợp có hình ảnh (equity curve, IC table, comparison chart)
- Ghi rõ trong README: nếu ML không đánh bại baseline ở phần nào, nói thẳng và giải thích tại sao — đây là điểm cộng, không phải điểm trừ

Deliverable cuối: GitHub repo hoàn chỉnh, sẵn sàng đưa vào CV/portfolio

---

## Bảng tóm tắt Layer ↔ Tuần

| Layer | Tuần tương ứng |
|---|---|
| Layer 1 — Methodology nền tảng | Tuần 1-2 |
| Layer 2 — Factor/alpha research | Tuần 4-5 (+ BRAIN song song từ tuần 4) |
| Layer 3 — ML/DL áp đúng chỗ | Tuần 3 (baseline), 6-7 (model) |
| Tổng hợp — backtest & báo cáo | Tuần 8-10 |

## Hoạt động song song, không tính vào timeline trên
- WorldQuant BRAIN: luyện viết alpha expression, đọc phần "Learn" của platform — bắt đầu từ tuần 4, duy trì đều đặn 1-2h/tuần
- Đọc "Advances in Financial Machine Learning" (chương liên quan) — rải rác trong suốt 10 tuần, không cần đọc hết 1 lần

## Rút gọn nếu cần intern sớm
Nếu cần gấp, có thể nén còn 5-6 tuần bằng cách:
- Gộp tuần 1-2 thành 1 tuần (bớt phần MA-crossover backtest thử nghiệm, đi thẳng vào data pipeline)
- Bỏ tuần 7 (LSTM) — LightGBM + baseline đã đủ thể hiện đúng phương pháp
- Rút backtest (tuần 8) xuống 3-4 ngày, chỉ cần 1 chiến lược đơn giản chạy đúng
