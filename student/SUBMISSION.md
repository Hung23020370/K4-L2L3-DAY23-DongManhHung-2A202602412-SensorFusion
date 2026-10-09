# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Đồng Mạnh Hùng
- MSSV: 2A202602412
- Email: donghung729@gmail.com
- Link repo (fork): https://github.com/Hung23020370/K4-L2L3-DAY23-DongManhHung-2A202602412-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): 25e3f1c397e8533342a7e864e6a5578f1c45756b

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: [0, 198] (199 frame); `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`; `seed`: 0
- Detection: `tp = 519`, `fp = 16`, `fn = 222`, `precision = 0.9701`, `recall = 0.7004`

| Chỉ số tracking | LiDAR-only | Fused (LiDAR + camera) |
|---|---|---|
| `rmse` (m) | 0.1503 | 0.1359 |
| `matches` | 502 | 502 |
| `sum_sq_err` | 11.344 | 9.267 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 239 | 239 |
| `mean_confirmed_tracks` | 2.523 | 2.523 |

Chỉ số dẫn xuất (theo công thức RUBRIC): `precision_track = 502 / (502 + 0) = 1.0`; `coverage = 502 / 519 = 0.967` (≥ 0.70); RMSE cả hai mode ≤ 0.45 m; `rmse_fused − rmse_lidar = −0.0145 m` (≤ 0.05 m, fused tốt hơn LiDAR).

### Giải thích khác biệt hai mode

Hai mode ghép **cùng số cặp (502)**, **cùng 0 ghost** và **cùng 239 miss**, nên so sánh RMSE là công bằng: cùng một tập cặp track–GT, không mode nào "đẹp hơn" nhờ ghép ít hơn. Ghost, miss và `mean_confirmed_tracks` (2.523) giống hệt nhau là đúng thiết kế, vì vòng đời track (khởi tạo, tăng giảm score, xác nhận, xoá) chỉ do LiDAR quyết định; camera chỉ chạy EKF update nên không thể thêm hay làm mất track.

Khác biệt nằm ở độ chính xác vị trí: fused có RMSE 0.1359 m so với 0.1503 m của LiDAR (giảm khoảng 9.6%), và `sum_sq_err` giảm từ 11.344 xuống 9.267 (khoảng 18%). Camera cho một ràng buộc góc (u, v) bổ sung cho đo LiDAR, nên cải thiện vị trí trong lúc track được cập nhật liên tục. Camera không có độ sâu, nên mình cho rằng phần cải thiện chủ yếu là theo hướng ngang, nhưng mình chưa tách sai số theo trục để chứng minh điều này. Cũng cần lưu ý kết quả phụ thuộc vào khoảng frame: ở lần chạy thử 21 frame (0–20), fused lại kém LiDAR khoảng 0.04 m (0.154 so với 0.114), nên không nên kết luận chung rằng fusion luôn tốt hơn.

Về miss: `valid_gt = matches + misses = 502 + 239 = 741`, đúng bằng `det_tp + det_fn = 519 + 222`. Trong 239 frame miss, **222 là do detector có sẵn bỏ sót** (recall 0.70), không phải do tracker. Phần còn lại 17 (= 519 − 502) là các xe detector đã phát hiện nhưng chưa có confirmed track ghép được trong cổng 2 m; nhiều khả năng do độ trễ xác nhận của track mới (cần score > 0.8) và có thể một số lần gán không qua gate. Chỉ số `coverage = 0.967` phản ánh đúng việc tracker bám gần hết số xe mà detector đã thấy, và không bị trừ cho phần detector bỏ sót.

**Giới hạn:** camera dùng tâm hộp 2D của nhãn ground-truth cộng nhiễu seeded, không phải camera detector; LiDAR detector là bộ có sẵn. Vì vậy kết quả không đo hiệu quả của một hệ thống perception độc lập với GT, và việc fused tốt hơn LiDAR ở đây một phần có thể do camera "biết trước" nhãn thật, chưa chắc áp dụng được cho camera detector thực tế.

## Giải thích ngắn (Parts E–H)

**1. Khác biệt đo LiDAR 3D và camera 2D trong EKF (`z`, `R`)?**
LiDAR đo vị trí tâm xe 3D: `z` có 3 phần tử (x, y, z) trong frame sensor, `R` là ma trận 3×3, và `h(x)` gần như là phép đổi hệ toạ độ tuyến tính nên Jacobian `H` gần hằng. Camera đo pixel 2D: `z = [u, v]`, `R = diag(σ_u², σ_v²)` (đơn vị pixel²), và `h(x)` phi tuyến theo mô hình pinhole: `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (xem `camera_measurement_prediction` trong `camera_fusion.py`), với `(x_s, y_s, z_s)` là vị trí trong frame camera. Vì phi tuyến nên EKF phải tuyến tính hoá bằng Jacobian tại trạng thái dự đoán. Phép chia cho độ sâu `x_s` làm camera mất thông tin khoảng cách (chỉ ràng buộc hướng nhìn), nên camera chỉ tinh chỉnh chứ không thay được LiDAR; code cũng chặn trường hợp `x_s ≤ 1e-6`.

**2. Vì sao cần gating Mahalanobis trước khi gán?**
Khoảng cách Euclid không tính đến độ bất định: cùng 1 m lệch nhưng track chắc chắn thì đáng ngờ, track bất định lớn thì bình thường. Bình phương khoảng cách Mahalanobis `d² = γᵀ S⁻¹ γ` với `S = H P Hᵀ + R` chuẩn hoá theo độ bất định đó và theo từng sensor. So với ngưỡng chi-square `chi2.ppf(gating_threshold, df = dim_meas)` (df = 3 cho LiDAR, 2 cho camera) trong hàm `chi2_gate`, ta loại các cặp không hợp lý trước khi tối ưu gán, tránh việc một đo lường sai (nhiễu, xe khác) được ghép nhầm và kéo lệch trạng thái EKF. Trong `association_cost_matrix`, kiểm tra `in_fov` làm trước, nên track ngoài tầm nhìn không bao giờ đi vào phép chiếu; cặp bị loại gán chi phí `inf`.

**3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log.**
Là **track-then-fuse** theo định nghĩa của lab (không gộp dữ liệu thô trước detection): chỉ có **một** tracker, giữ một danh sách track, EKF predict **một lần** mỗi frame, rồi lần lượt update theo từng modality sau bước gán riêng: gán LiDAR (AssocL) rồi gán camera (AssocC) khi bật fusion. Trong `run_lab.py`, mỗi frame chạy `KF.predict` cho mọi track, sau đó `associate_and_update(..., lidar_sensor)`, rồi (chỉ ở mode fused) `associate_and_update(..., camera_sensor)` trên cùng `manager.track_list`. Mỗi sensor đã qua detection và gán riêng của nó trước khi cập nhật vào track chung; không có bước trộn dữ liệu thô (point cloud và ảnh) trước detection. Trên log `grade_run.log`, mỗi `(mode, frame)` chỉ có một bộ `confirmed/matches/ghosts/misses` cho mode `fused` (và một cho `lidar`), không có track riêng theo sensor; ghost/miss/confirmed của fused trùng với lidar cho thấy chỉ có một tập track do LiDAR quyết định vòng đời.

**4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
Innovation `γ = z − h(x)` sẽ có độ lệch hệ thống (trung bình khác 0, không còn là nhiễu trắng quanh 0), thường lệch theo trục u nếu sai góc yaw và theo v nếu sai pitch hoặc độ cao, và tăng theo khoảng cách. `d²` tăng, nên tỉ lệ cặp bị gating chặn tăng. Có hai trường hợp: lệch lớn thì camera measurement bị gating loại, fused gần như thành LiDAR-only; lệch vừa phải thì vẫn lọt gate và kéo state lệch có hệ thống, làm RMSE fused tăng. Trong lần chạy hiện tại fused có RMSE thấp hơn LiDAR (0.136 so với 0.150 m) nên không thấy dấu hiệu lệch calibration; nhưng nếu extrinsic bị lệch, mình kỳ vọng RMSE fused tăng lên hoặc camera update bị gating loại dần.

**5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì sao LiDAR quyết định score/init/delete còn camera chỉ EKF update?**
Khi `meas_list` rỗng thì không có phép đo nào mang theo thông tin `sensor`, nên không thể suy ra đây là lượt LiDAR hay camera. Hàm vẫn luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` ở cuối. Với LiDAR, frame rỗng vẫn phải trừ score các track miss trong FOV, xoá track yếu và sinh track mới; nếu bỏ qua, track ma sẽ sống mãi. Với camera, frame rỗng hoặc camera miss thì không được phạt hay xoá track gì cả. Vì thế `sensor` phải truyền tường minh để phân biệt hai trường hợp. LiDAR quyết định vòng đời vì chỉ LiDAR cho vị trí 3D đủ để khởi tạo track (camera không có độ sâu) và có độ tin cậy cao; camera có FOV hẹp hơn nên không thấy cũng không có nghĩa là xe biến mất; và nếu camera cũng tăng score thì cùng một frame bị đếm hai lần, làm score không còn là "một lần cập nhật mỗi frame LiDAR". Vậy camera chỉ gọi EKF update để tinh chỉnh state.

**6. Điều kiện xác nhận, giữ confirmed sau miss, và xoá track?**
Theo `track_management.py`: track mới sinh từ đo LiDAR có `state = "initialized"` và `score = 1/window`. Mỗi lần LiDAR hit, `score += 1/window` (tối đa 1); khi `score > confirmed_threshold` thì track thành `confirmed`, nếu chưa thì là `tentative`. Mỗi lần LiDAR miss trong FOV, `score −= 1/window`, nhưng trạng thái đã `confirmed` **không bị hạ về tentative** (giữ confirmed sau miss). Xoá track khi: (a) `P[0,0]` hoặc `P[1,1]` vượt `max_P` (bất định ngang quá lớn), bất kể score; (b) track confirmed có `score < delete_threshold`; (c) track chưa confirmed có `score ≤ 0`. Giá trị trong `tracking_params.py`: `window = 6` (mỗi hit/miss đổi score 1/6), `confirmed_threshold = 0.8`, `delete_threshold = 0.6`, `max_P = 3.0² = 9`, `gating_threshold = 0.995`. Ví dụ: track mới có score 1/6 ≈ 0.17, cần score > 0.8 (tức lần sinh track cộng thêm ít nhất 4 lần hit LiDAR (1/6 + 4/6 ≈ 0.83)) mới thành confirmed, nên các frame đầu của một xe thường chưa có confirmed track (nhiều khả năng là nguồn của 17 frame detector thấy nhưng chưa có confirmed track ghép được). Sau khi đã confirmed, track chỉ bị xoá khi score < 0.6, tức sau vài lần miss liên tiếp trong FOV LiDAR.

## Bonus (không bắt buộc)

Không.

## Khai báo sử dụng AI (bắt buộc)


- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Anthropic).
- Dùng cho phần nào: debug lỗi chạy CP5 (`FileNotFoundError` do tên segment `.tfrecord` trong `paths.yaml` không khớp file đã tải) và hỗ trợ soạn thảo báo cáo này từ `metrics.json` và code của mình.
- Cách đã kiểm tra lại: `pytest student/tests -q` không còn failed/xfailed; chạy `fusion-run-lab --fusion compare --seed 0` trên Waymo; đối chiếu số liệu trong báo cáo với `metrics.json` và `grade_run.log`; đối chiếu lời giải thích với code trong `workspace/`.

## Checklist nộp

- [x] Part E–H trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS
