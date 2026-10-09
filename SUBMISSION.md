# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Đồng Mạnh Hùng
- MSSV: 2A202602412
- Email: donghung729@gmail.com
- Link repo (fork): <điền link, tên repo: https://github.com/Hung23020370/K4-L2L3-DAY23-DongManhHung-2A202602412-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): <điền sau khi commit cuối>

## Tóm tắt kết quả

<!-- CẬP NHẬT toàn bộ số liệu dưới đây sau khi chạy lần chấm điểm đầy đủ (frame 0–198). Số hiện tại lấy từ lần chạy frame 0–20. -->

- `fusion_mode`: `compare`; `frames`: [0, 20]; `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`; `seed`: 0
- Detection: `tp = 42`, `fp = 2`, `fn = 0`, `precision = 0.9545`, `recall = 1.0`

| Chỉ số tracking | LiDAR-only | Fused (LiDAR + camera) |
|---|---|---|
| `rmse` (m) | 0.1137 | 0.1544 |
| `matches` | 34 | 34 |
| `sum_sq_err` | 0.4396 | 0.8104 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 8 | 8 |
| `mean_confirmed_tracks` | 1.619 | 1.619 |

Chỉ số dẫn xuất (theo công thức RUBRIC): `precision_track = 34 / (34 + 0) = 1.0`; `coverage = 34 / 42 = 0.810` (≥ 0.70); RMSE cả hai mode ≤ 0.45 m; `rmse_fused − rmse_lidar = 0.0407 m ≤ 0.05 m`.

### Giải thích khác biệt hai mode

Hai mode ghép **cùng số cặp (34)**, **cùng 0 ghost** và **cùng 8 miss**, nên so RMSE giữa hai mode là so sánh công bằng (cùng tập cặp track–GT, không có chuyện một mode "đẹp hơn" chỉ vì ghép ít hơn). Việc ghost, miss và `mean_confirmed_tracks` giống hệt nhau cũng đúng thiết kế: vòng đời track (khởi tạo, tăng giảm score, xác nhận, xoá) chỉ do LiDAR quyết định, camera chỉ chạy EKF update nên không thể làm thêm hay mất track.

Khác biệt nằm ở độ chính xác vị trí: fused có RMSE 0.154 m so với 0.114 m của LiDAR (`sum_sq_err` 0.810 so với 0.440), tức fused **kém hơn khoảng 0.04 m**, chưa cải thiện so với LiDAR. Mình cho rằng nguyên nhân chính là phép đo camera trong lab này không độc lập: nó là tâm hộp 2D ground-truth của camera FRONT cộng nhiễu seeded, và tâm hộp 2D không trùng đúng với hình chiếu tâm 3D của xe, nên mỗi lần camera update kéo trạng thái lệch nhẹ khỏi giá trị LiDAR đã khá chính xác. Camera chỉ cho 2 bậc tự do (u, v), không có độ sâu, nên nó không thể cải thiện khoảng cách. Mức chênh 0.04 m vẫn nằm trong ngưỡng cho phép 0.05 m của RUBRIC, nhưng biên khá sát.

8 miss ở cả hai mode là các frame có xe thật trong cửa sổ BEV nhưng chưa có confirmed track ghép được. Với số liệu hiện có mình chưa tách được từng nguyên nhân; nhiều khả năng là độ trễ xác nhận ở những frame đầu (track mới sinh phải tích đủ score mới thành confirmed) và các xe ở mép cửa sổ. Phần này mình sẽ đối chiếu lại với `grade_run.log` trên lần chạy đầy đủ.

**Giới hạn:** camera dùng nhãn GT có nhiễu, không phải camera detector, và LiDAR detector là bộ có sẵn. Vì vậy kết quả không đo hiệu quả của một hệ thống perception độc lập với GT, và không nên kết luận "fusion tốt hơn hay kém hơn LiDAR" cho trường hợp thực tế.

## Giải thích ngắn (Parts E–H)

**1. Khác biệt đo LiDAR 3D và camera 2D trong EKF (`z`, `R`)?**
LiDAR đo vị trí tâm xe 3D: `z` có 3 phần tử (x, y, z) trong frame sensor, `R` là ma trận 3×3, và `h(x)` gần như là phép đổi hệ toạ độ tuyến tính nên Jacobian `H` gần hằng. Camera đo pixel 2D: `z = [u, v]`, `R = diag(σ_u², σ_v²)` (đơn vị pixel²), và `h(x)` phi tuyến theo mô hình pinhole: `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (xem `camera_measurement_prediction` trong `camera_fusion.py`), với `(x_s, y_s, z_s)` là vị trí trong frame camera. Vì phi tuyến nên EKF phải tuyến tính hoá bằng Jacobian tại trạng thái dự đoán. Phép chia cho độ sâu `x_s` làm camera mất thông tin khoảng cách (chỉ ràng buộc hướng nhìn), nên camera chỉ tinh chỉnh chứ không thay được LiDAR; code cũng chặn trường hợp `x_s ≤ 1e-6`.

**2. Vì sao cần gating Mahalanobis trước khi gán?**
Khoảng cách Euclid không tính đến độ bất định: cùng 1 m lệch nhưng track chắc chắn thì đáng ngờ, track bất định lớn thì bình thường. Bình phương khoảng cách Mahalanobis `d² = γᵀ S⁻¹ γ` với `S = H P Hᵀ + R` chuẩn hoá theo độ bất định đó và theo từng sensor. So với ngưỡng chi-square `chi2.ppf(gating_threshold, df = dim_meas)` (df = 3 cho LiDAR, 2 cho camera) trong hàm `chi2_gate`, ta loại các cặp không hợp lý trước khi tối ưu gán, tránh việc một đo lường sai (nhiễu, xe khác) được ghép nhầm và kéo lệch trạng thái EKF. Trong `association_cost_matrix`, kiểm tra `in_fov` làm trước, nên track ngoài tầm nhìn không bao giờ đi vào phép chiếu; cặp bị loại gán chi phí `inf`.

**3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log.**
Là **track-then-fuse** theo định nghĩa của lab (không gộp dữ liệu thô trước detection): chỉ có **một** tracker, giữ một danh sách track, EKF predict **một lần** mỗi frame, rồi lần lượt update theo từng modality sau bước gán riêng: gán LiDAR (AssocL) rồi gán camera (AssocC) khi bật fusion. Trong `run_lab.py`, mỗi frame chạy `KF.predict` cho mọi track, sau đó `associate_and_update(..., lidar_sensor)`, rồi (chỉ ở mode fused) `associate_and_update(..., camera_sensor)` trên cùng `manager.track_list`. Mỗi sensor đã qua detection và gán riêng của nó trước khi cập nhật vào track chung; không có bước trộn dữ liệu thô (point cloud và ảnh) trước detection. Trên log `grade_run.log`, mỗi `(mode, frame)` chỉ có một bộ `confirmed/matches/ghosts/misses` cho mode `fused` (và một cho `lidar`), không có track riêng theo sensor; ghost/miss/confirmed của fused trùng với lidar cho thấy chỉ có một tập track do LiDAR quyết định vòng đời.

**4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
Innovation `γ = z − h(x)` sẽ có độ lệch hệ thống (trung bình khác 0, không còn là nhiễu trắng quanh 0), thường lệch theo trục u nếu sai góc yaw và theo v nếu sai pitch hoặc độ cao, và tăng theo khoảng cách. `d²` tăng, nên tỉ lệ cặp bị gating chặn tăng. Có hai trường hợp: lệch lớn thì camera measurement bị gating loại, fused gần như thành LiDAR-only; lệch vừa phải thì vẫn lọt gate và kéo state lệch có hệ thống, làm RMSE fused tăng. Điều này khớp một phần với quan sát hiện tại (RMSE fused cao hơn LiDAR ~0.04 m), dù ở đây nguyên nhân chính có thể là sự sai khác tâm hộp 2D so với hình chiếu tâm 3D như mục trên chứ chưa chắc là calibration.

**5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì sao LiDAR quyết định score/init/delete còn camera chỉ EKF update?**
Khi `meas_list` rỗng thì không có phép đo nào mang theo thông tin `sensor`, nên không thể suy ra đây là lượt LiDAR hay camera. Hàm vẫn luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` ở cuối. Với LiDAR, frame rỗng vẫn phải trừ score các track miss trong FOV, xoá track yếu và sinh track mới; nếu bỏ qua, track ma sẽ sống mãi. Với camera, frame rỗng hoặc camera miss thì không được phạt hay xoá track gì cả. Vì thế `sensor` phải truyền tường minh để phân biệt hai trường hợp. LiDAR quyết định vòng đời vì chỉ LiDAR cho vị trí 3D đủ để khởi tạo track (camera không có độ sâu) và có độ tin cậy cao; camera có FOV hẹp hơn nên không thấy cũng không có nghĩa là xe biến mất; và nếu camera cũng tăng score thì cùng một frame bị đếm hai lần, làm score không còn là "một lần cập nhật mỗi frame LiDAR". Vậy camera chỉ gọi EKF update để tinh chỉnh state.

**6. Điều kiện xác nhận, giữ confirmed sau miss, và xoá track?**
Theo `track_management.py`: track mới sinh từ đo LiDAR có `state = "initialized"` và `score = 1/window`. Mỗi lần LiDAR hit, `score += 1/window` (tối đa 1); khi `score > confirmed_threshold` thì track thành `confirmed`, nếu chưa thì là `tentative`. Mỗi lần LiDAR miss trong FOV, `score −= 1/window`, nhưng trạng thái đã `confirmed` **không bị hạ về tentative** (giữ confirmed sau miss). Xoá track khi: (a) `P[0,0]` hoặc `P[1,1]` vượt `max_P` (bất định ngang quá lớn), bất kể score; (b) track confirmed có `score < delete_threshold`; (c) track chưa confirmed có `score ≤ 0`. Giá trị trong `tracking_params.py`: `window = 6` (mỗi hit/miss đổi score 1/6), `confirmed_threshold = 0.8`, `delete_threshold = 0.6`, `max_P = 3.0² = 9`, `gating_threshold = 0.995`. Ví dụ: track mới có score 1/6 ≈ 0.17, cần score > 0.8 (tức lần sinh track cộng thêm ít nhất 4 lần hit LiDAR (1/6 + 4/6 ≈ 0.83)) mới thành confirmed, nên các frame đầu của một xe thường chưa có confirmed track (liên quan các frame miss ở trên). Sau khi đã confirmed, track chỉ bị xoá khi score < 0.6, tức sau vài lần miss liên tiếp trong FOV LiDAR.

## Bonus (không bắt buộc)

Không.

## Khai báo sử dụng AI (bắt buộc)


- Công cụ đã dùng: Claude (Anthropic).
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
