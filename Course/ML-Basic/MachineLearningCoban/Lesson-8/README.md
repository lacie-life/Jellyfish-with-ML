## Logistic Regression

Tài liệu tham khảo: https://machinelearningcoban.com/2017/01/27/logisticregression/

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/1.PNG?raw=true)

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/2.PNG?raw=true)

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/3.PNG?raw=true)

- Đường màu vàng biểu diễn linear regression. Đường này không bị chặn nên không phù hợp cho bài toán này. Có một trick nhỏ để đưa nó về dạng bị chặn: cắt phần nhỏ hơn 0 bằng cách cho chúng bằng 0, cắt các phần lớn hơn 1 bằng cách cho chúng bằng 1. Sau đó lấy điểm trên đường thẳng này có tung độ bằng 0.5 làm điểm phân chia hai class, đây cũng không phải là một lựa chọn tốt. Giả sử có thêm vài bạn sinh viên tiêu biểu ôn tập đến 20 giờ và, tất nhiên, thi đỗ. Khi áp dụng mô hình linear regression như hình dưới đây và lấy mốc 0.5 để phân lớp, toàn bộ sinh viên thi trượt vẫn được dự đoán là trượt, nhưng rất nhiều sinh viên thi đỗ cũng được dự đoán là trượt (nếu ta coi điểm x màu xanh lục là ngưỡng cứng để đưa ra kết luận). Rõ ràng đây là một mô hình không tốt. Anh chàng sinh viên tiêu biểu này đã kéo theo rất nhiều bạn khác bị trượt.

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/ex1_lr.png?raw=true)

- Đường màu đỏ (chỉ khác với activation function của PLA ở chỗ hai class là 0 và 1 thay vì -1 và 1) cũng thuộc dạng ngưỡng cứng (hard threshold). PLA không hoạt động trong bài toán này vì dữ liệu đã cho không linearly separable.
- Các đường màu xanh lam và xanh lục phù hợp với bài toán của chúng ta hơn. Chúng có một vài tính chất quan trọng sau:

+ Là hàm số liên tục nhận giá trị thực, bị chặn trong khoảng (0,1).
+ Nếu coi điểm có tung độ là 1/2 làm điểm phân chia thì các điểm càng xa điểm này về phía bên trái có giá trị càng gần 0. Ngược lại, các điểm càng xa điểm này về phía phải có giá trị càng gần 1. Điều này khớp với nhận xét rằng học càng nhiều thì xác suất đỗ càng cao và ngược lại.
+ Mượt (smooth) nên có đạo hàm mọi nơi, có thể được lợi trong việc tối ưu.

### Sigmoid function

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/4.PNG?raw=true)

## Hàm mất mát và phương pháp tối ưu (đoạn này hơi nhiều toán, mình cũng k hiểu nhưng mà thấy nó hay nên đưa vảo, tôn trọng vẻ đẹp của toán học chứ nhề)

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/5.PNG?raw=true)
![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/6.PNG?raw=true)
![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/7.PNG?raw=true)
![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/8.PNG?raw=true)

### Chốt :))))
![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/9.PNG?raw=true)

Có example Python nha, vào link tài liệu ý

## Ví dụ với dữ liệu 2 chiều

Chúng ta xét thêm một ví dụ nhỏ nữa trong không gian hai chiều. Giả sử chúng ta có hai class xanh-đỏ với dữ liệu được phân bố như hình dưới.

![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/logistic_2d.png?raw=true)

Với dữ liệu đầu vào nằm trong không gian hai chiều, hàm sigmoid có dạng như thác nước dưới đây:

![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/plaszczyzna.gif?raw=true)

Kết quả tìm được khi áp dụng mô hình logistic regression được minh họa như hình dưới với màu nền khác nhau thể hiện xác suất điểm đó thuộc class đỏ. Đỏ hơn tức gần 1 hơn, xanh hơn tức gần 0 hơn.

![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/logistic_2d_2.png?raw=true)

Nếu phải lựa chọn một ngưỡng cứng (chứ không chấp nhận xác suất) để phân chia hai class, chúng ta quan sát thấy đường thẳng nằm nằm trong khu vực xanh lục là một lựa chọn hợp lý. Tôi sẽ chứng minh ở phần dưới rằng, đường phân chia giữa hai class tìm được bởi logistic regression có dạng một đường phẳng, tức vẫn là linear.

## Một vài tính chất của Logistic Regression

- Logistic Regression thực ra được sử dụng nhiều trong các bài toán Classification.

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/10.PNG?raw=true)

- Boundary tạo bởi Logistic Regression có dạng tuyến tính

## Tác giả thêm chút thảo luận nè

![Hình 15](https://github.com/lacie-life/ML-basic/blob/master/Lesson8/img/11.PNG?raw=true)

/////////////////Bài này nhiều toán vãi :)))))))))))))////////////////////////