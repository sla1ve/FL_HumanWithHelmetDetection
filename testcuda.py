import torch

# Kiểm tra CUDA có khả dụng không
cuda_available = torch.cuda.is_available()
print(f"CUDA có khả dụng: {cuda_available}")

# Kiểm tra tên của thiết bị CUDA (nếu có)
if cuda_available:
    print(f"Tên thiết bị: {torch.cuda.get_device_name(0)}")
    print(f"Tổng số thiết bị: {torch.cuda.device_count()}")
else:
    print("Không có thiết bị CUDA nào khả dụng.")