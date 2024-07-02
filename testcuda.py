import torch

# Kiểm tra CUDA có khả dụng không
cuda_available = torch.cuda.is_available()
print("CUDA có khả dụng:", "✓" if cuda_available else "✗")

# Kiểm tra tên của thiết bị CUDA (nếu có)
if cuda_available:
    print("Tên thiết bị:", torch.cuda.get_device_name(0))
    print("Tổng số thiết bị:", torch.cuda.device_count())
else:
    print("Không có thiết bị CUDA nào khả dụng.")
