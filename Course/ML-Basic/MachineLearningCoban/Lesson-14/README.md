##  Convex sets và convex functions

Tài liệu này: chứ bài này thấy bảo toán là vl rồi : https://machinelearningcoban.com/2017/03/12/convexity/

Định nghĩa 1: Một tập hợp được gọi là tập lồi (convex set) nếu đoạn thẳng nối hai điểm bất kỳ trong tập hợp hợp đó nằm trọn vẹn trong tập hợp đó.

Machine Learning và Optimization có quan hệ mật thiết với nhau. Trong Optimization, Convex Optimization là quan trọng nhất. Một bài toán là convex optimization nếu hàm mục tiêu là convex và tập hợp các điểm thỏa mãn các điều kiện ràng buộc là một convex set.

Trong convex set, mọi đoạn thẳng nối hai điểm bất kỳ trong tập đó sẽ nằm hoàn toàn trong tập đó. Tập hợp các giao điểm của các convex sets là một convex set.

Một hàm số là convex nếu đoạn thẳng nối hai điểm bất kỳ trên đồ thì hàm số đó không nằm dưới đồ thị đó.

Một hàm số khả vi là convex nếu tập xác định của nó là convex và đường (mặt) tiếp tuyến không nằm phía trên đồ thị (bề mặt) của hàm số đó.

Các norms là các hàm lồi, được sử dụng nhiều trong tối ưu.

Bài này chịu

----------------------------------------------------------------------------------------------------------------------------------------
## Convex Optimization Problems

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/1.PNG?raw=true)

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/2.PNG?raw=true)

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/3.PNG?raw=true)

### Bài toán tối ưu lồi

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/4.PNG?raw=true)

### Cực tiểu của bài toán tối ưu lồi chính là điểm tối ưu.

TÍnh chất quan trọng nhất của bài toán tối ưu lồi chính là bất kỳ locally optimal point chính là một điểm (globally) optimal point.

Sau đấy còn 1 đống toán nữa mà mình chịu, méo hiểu đc. Tâm trạng cũng đang đéo tốt chút nào . ĐM

## Chốt
Các bài toán tối ưu xuất hiện rất nhiều trong thực tế, trong đó Tối Ưu Lồi đóng một vai trò quan trọng. Trong bài toán Tối Ưu Lồi, nếu tìm được cực trị thì cực trị đó chính là một điểm optimal của bài toán (nghiệm của bài toán).

Có nhiều bài toán tối ưu không được viết dưới dạng convex nhưng có thể biến đổi về dạng convex, ví dụ như bài toán Geometric Programming.

Linear Programming và Quadratic Programming đóng một vài trò quan trọng trong toán tối ưu, được sử dụng nhiều trong các thuật toán Machine Learning.

Thư viện CVXOPT được dùng để tối ưu nhiều bài toán tối ưu lồi, rất dễ sử dụng và thời gian chạy tương đối nhanh.

-------------------------------------------------------------------------------------------------------------------------------------------------

## Duality

### Giới thiệu

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/5.PNG?raw=true)

### Phương pháp nhân tử Lagrange

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/6.PNG?raw=true)

![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/7.PNG?raw=true)

![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/8.PNG?raw=true)

### Hàm đối ngẫu Lagrange (The Lagrange dual function)

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/9.PNG?raw=true)

### Hàm đối ngẫu của một bài toán tối ưu bất kỳ là một hàm concave, bất kể bài toán ban đầu có phải là convex hay không.

![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/10.PNG?raw=true)

![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/11.PNG?raw=true)

![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/12.PNG?raw=true)

![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/13.PNG?raw=true)

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/14.PNG?raw=true)

### Bài toán đối ngẫu Lagrange (The Lagrange dual problem)

![Hình 15](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/15.PNG?raw=true)

![Hình 16](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/16.PNG?raw=true)

![Hình 17](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/17.PNG?raw=true)

### Optimality conditions

![Hình 18](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/18.PNG?raw=true)

![Hình 19](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/19.PNG?raw=true)

![Hình 20](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/20.PNG?raw=true)

![Hình 21](https://github.com/lacie-life/ML-basic/blob/master/Lesson14/img/21.PNG?raw=true)



