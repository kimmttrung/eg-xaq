# Hướng dẫn gán nhãn episode

Bộ episode là căn cứ cho Cause F1 và bảng ablation A–E — số liệu chính của khóa luận.
Nhãn sai hoặc không có nguồn thì mọi con số phía sau đều vô nghĩa.

## Nguyên tắc duy nhất không được vi phạm

**Nhãn đến từ nguồn độc lập, không đến từ hệ thống.**

- Không chạy `demo_explain.py` cho ngày đó rồi chép kết quả vào nhãn.
- Không nhìn PBLH / gió / mưa của ngày đó rồi tự suy ra cơ chế. Làm vậy là tái tạo
  lại rule, và Cause F1 chỉ đo hệ thống có khớp với chính nó không. Phiếu gán nhãn cố
  ý không in các số này.
- Chỉ ghi cơ chế khi nguồn **nói ra**, bằng lời của nguồn.

## Quy trình

```powershell
# 1. Sinh phiếu: chọn ngày theo PM2.5 quan trắc
python scripts/select_episode_candidates.py --csv ../era5/dataset_train.csv

# 2. Gán nhãn từng mục trong data/eval/episodes.yaml (xem bên dưới)

# 3. Kiểm tra file hợp lệ + chấm điểm
python scripts/score_episodes.py
```

Với mỗi ngày trong phiếu:

1. **Tìm nguồn** theo thứ tự ưu tiên (bảng dưới). Tìm theo ngày ±2 ngày — báo cáo
   thường mô tả cả đợt, không từng ngày.
2. **Đọc** nguồn nói nguyên nhân là gì.
3. **Đối chiếu** với danh sách cơ chế ở đầu file `episodes.yaml`, xếp vào ba danh sách.
4. **Điền** `rationale`, `sources`, `annotator`, `label_confidence`.
5. **Đổi** `status: todo` → `labeled`.
6. Không tìm được nguồn sau khoảng 20 phút → `status: rejected` + `reject_reason`.
   **Loại bỏ là kết quả hợp lệ**, tốt hơn nhiều so với đoán. Tỉ lệ bị loại cũng được báo cáo.

## Nguồn nhãn — thứ tự ưu tiên

| `type` | Nguồn | Ghi chú |
|---|---|---|
| `cem_report` | Báo cáo / bản tin của CEM, Sở TN&MT Hà Nội | Ưu tiên cao nhất |
| `peer_reviewed` | Bài báo phân tích **đúng đợt đó** | Bài nói chung về Hà Nội không tính. Corpus có 32 bài về Việt Nam — xem `python scripts/corpus_coverage.py` |
| `forecast_bulletin` | Bản tin dự báo chất lượng không khí | |
| `news_expert` | Báo chí **có dẫn tên chuyên gia/cơ quan** | Bài báo không dẫn nguồn → không tính |
| `supervisor` | Ý kiến GVHD | Phương án cuối. Tỉ lệ episode chỉ dựa vào nguồn này được báo cáo riêng |

`locator`: URL, DOI, hoặc số trang / mục — đủ để người khác tìm lại đúng chỗ.

## Ba danh sách cơ chế

| Danh sách | Khi nào | Hệ thống bị chấm thế nào |
|---|---|---|
| `primary_mechanisms` | Nguồn nói **rõ** là nguyên nhân chính | Bỏ sót = lỗi |
| `contributing_mechanisms` | Nguồn nhắc tới như yếu tố góp phần | Nêu ra: không phạt ở F1 nới, bị tính dương tính giả ở F1 chặt |
| `excluded_mechanisms` | Nguồn **loại trừ** rõ ràng ("không phải do đốt rơm rạ") | Khẳng định = lỗi nặng nhất, đếm riêng |

Cơ chế nguồn không nhắc tới → **không ghi vào đâu cả**. Đừng tự thêm vào `excluded` chỉ
vì nghĩ là không liên quan.

## `expected_outcome`

| Giá trị | Khi nào | `primary_mechanisms` |
|---|---|---|
| `cause` | Ngày ô nhiễm, nguồn nêu nguyên nhân | ≥1 cơ chế làm **tăng** (↑) |
| `clean` | Ngày sạch, nguồn nêu vì sao sạch | chỉ cơ chế làm **giảm** (↓) |
| `insufficient` | Nguồn xác nhận ngày đó nhưng nói không rõ nguyên nhân | để trống |

## `label_confidence`

- `high` — nguồn ưu tiên cao, nói thẳng nguyên nhân, đúng ngày.
- `medium` — nguồn mô tả cả đợt, hoặc diễn đạt gián tiếp cần suy luận ít.
- `low` — chỉ có `news_expert` / `supervisor`, hoặc nguồn mơ hồ.

## Ví dụ một mục đã gán nhãn

Giá trị trong ví dụ là **minh họa định dạng**, không phải nhãn thật.

```yaml
  - episode_id: hn-2023-01-12
    status: labeled
    date: 2023-01-12
    group: polluted_winter
    pm25_obs_ugm3: 118
    expected_outcome: cause
    questions:
      - Vì sao hôm nay Hà Nội ô nhiễm nặng?
      - Có phải do đốt rơm rạ không?
    primary_mechanisms: [MECH_INVERSION, MECH_STAGNATION]
    contributing_mechanisms: [MECH_NO_WET_REMOVAL]
    excluded_mechanisms: [MECH_BIOMASS_TRANSPORT]
    rationale: "Bản tin nêu hiện tượng nghịch nhiệt và gió yếu kéo dài, không mưa; khẳng định không ghi nhận điểm cháy lớn quanh Hà Nội."
    sources:
      - type: cem_report
        citation: "<tên bản tin>, <cơ quan>, <ngày phát hành>"
        locator: "<URL hoặc số trang>"
    annotator: "<tên bạn>"
    label_confidence: high
```

Mục bị loại:

```yaml
  - episode_id: hn-2022-10-03
    status: rejected
    reject_reason: "Không tìm được báo cáo hay bài báo nào về đợt này sau khi tra CEM và báo chí ±2 ngày"
```

## Trước khi báo cáo số liệu

- [ ] Hỏi mentor (M10): các ngày có nằm trong **tập huấn luyện** của mô hình lab không.
  Nếu có, SHAP đang giải thích một dự báo đã "thấy đáp án" (docs/06 §2.3).
- [ ] Người thứ hai gán nhãn độc lập ít nhất 10 episode → tính Cohen's κ.
- [ ] Báo cáo: số `labeled` / `rejected`, tỉ lệ `supervisor`, phân bố `label_confidence`.
- [ ] Nguồn PM2.5 dùng để chọn ngày (OpenAQ, trạm nào) ghi vào chương phương pháp.
