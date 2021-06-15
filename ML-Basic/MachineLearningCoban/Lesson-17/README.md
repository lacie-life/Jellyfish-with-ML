# Kernel Support Vector Machine

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/1.PNG?raw=true)

Xét ví dụ dưới đây với việc biến dữ liệu không phân biệt tuyến tính trong không gian hai chiều thành phân biệt tuyến tính trong không gian ba chiều bằng cách giới thiệu thêm một chiều mới:

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/2.PNG?raw=true)

Youtube: https://www.youtube.com/watch?v=04eOsL5vrWc

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/3.PNG?raw=true)

## Cơ sở toán học

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/4.PNG?raw=true)

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/5.PNG?raw=true)

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/6.PNG?raw=true)

## Hàm số kernel

![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/7.PNG?raw=true)

![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/8.PNG?raw=true)

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson-17/img/9.PNG?raw=true)

## Ví dụ minh họa

read here: https://machinelearningcoban.com/2017/04/22/kernelsmv/

## Tóm tắt

+ Nếu dữ liệu của hai lớp là không phân biệt tuyến tính, chúng ta có thể tìm cách biến đổi dữ liệu sang một không gian mới sao cho trong không gian mới ấy, dữ liệu của hai lớp là phân biệt tuyến tính hoặc gần phân biệt tuyến tính.

+ Việc tính toán trực tiếp hàm Φ() đôi khi phức tạp và tốn nhiều bộ nhớ. Thay vào đó, ta có thể sử dụng kernel trick. Trong cách tiếp cận này, ta chỉ cần tính tích vô hướng của hai vector bất kỳ trong không gian mới: k(x,z)=Φ(x)TΦ(z).

+ Thông thường, các hàm k() thỏa mãn điều kiện Merrcer, và được gọi là kernel. Cách giải bài toán SVM với kernel hoàn toàn giống với cách giải bài toán Soft Margin SVM.

+ Có 4 loại kernel thông dụng: linear, poly, rbf, sigmoid. Trong đó, rbf được sử dụng nhiều nhất và là lựa chọn mặc định trong các thư viện SVM.

+ Với dữ liệu gần phân biệt tuyến tính, linear và poly kernels cho kết quả tốt hơn.