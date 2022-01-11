## Multi-layer Perceptron và Backpropagation

Tài liệu: https://machinelearningcoban.com/2017/02/24/mlp/

### Tổng quan Supervised Learning

Bài toán Supervised Learning, nói một cách ngắn gọn, là việc đi tìm một hàm số để với mỗi input, ta sử dụng hàm số đó để dự đoán output. Hàm số này được xây dựng dựa trên các cặp dữ liệu (xi,yi) trong training set. Nếu đầu ra dự đoán (predicted output) gần với đầu ra thực sự (ground truth) thì đó được gọi là một thuật toán tốt (nhưng khi đầu ra dự đoán quá giống với đầu ra thực sự thì không hẳn đã tốt)

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/1.PNG?raw=true)

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/2.PNG?raw=true)

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/3.PNG?raw=true)


### Các ký hiệu và khái niệm

## Layers

Ngoài Input layers và Output layers, một Multi-layer Perceptron (MLP) có thể có nhiều Hidden layers ở giữa. Các Hidden layers theo thứ tự từ input layer đến output layer được đánh số thứ thự là Hidden layer 1, Hidden layer 2, …

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/multi_layers.png?raw=true)

Số lượng layer trong một MLP được tính bằng số hidden layers cộng với 1. Tức là khi đếm số layers của một MLP, ta không tính input layers. Số lượng layer trong một MLP thường được ký hiệu là L. Trong hình trên đây, L=3.

## Units

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/4.PNG?raw=true)

## Weights và Biases

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/5.PNG?raw=true)

## Activation functions

![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/6.PNG?raw=true)

### Hàm sgn không được sử dụng trong MLP

Hàm sgn (còn gọi là hard-threshold) chỉ được sử dụng trong PLA, mang mục đích giáo dục nhiều hơn. Trong thực tế, hàm sgn không được sử dụng vì hai lý do: đầu ra là discrete, và đạo hàm tại hầu hết các điểm bằng 0 (trừ điểm 0 không có đạo hàm). Việc đạo hàm bằng 0 này khiến cho các thuật toán gradient-based (ví dụ như Gradient Descent) không hoạt động!

### Sigmoid và tanh

![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/7.PNG?raw=true)

### ReLU

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/8.PNG?raw=true)

### Một vài lưu ý

- Output layer nhiều khi không có activation function mà sử dụng trực tiếp giá trị đầu vào z(l)i của mỗi unit. Hoặc nói một cách khác, activation function chính là hàm identity, tức đầu ra bằng đầu vào. Với các bài toán classification, output layer thường là một Softmax Regression layer giúp tính xác suất để một điểm dữ liệu rơi vào mỗi class.

- Mặc dù activation function cho mỗi unit có thể khác nhau, trong cùng một network, activation như nhau thường được sử dụng. Điều này giúp cho việc tính toán được đơn giản hơn.

## Backpropagation

![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/9.PNG?raw=true)

![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/10.PNG?raw=true)

![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/11.PNG?raw=true)

![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/12.PNG?raw=true)

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/13.PNG?raw=true)

![Hình 15](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/14.PNG?raw=true)

![Hình 16](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/15.PNG?raw=true)

![Hình 17](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/16.PNG?raw=true)

Một cái gì đó rất pro : https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.g76s9xxzc

## Thảo luận 

![Hình 18](https://github.com/lacie-life/ML-basic/blob/master/Lesson12/img/17.PNG?raw=true)


