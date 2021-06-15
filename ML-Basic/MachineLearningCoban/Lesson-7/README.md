## Perceptron Learning Algorithm (PLA)

### Chân lý: Cứ làm đi, sai đâu sửa đấy, cuối cùng cũng sẽ thành công

Tài liệu: https://machinelearningcoban.com/2017/01/21/perceptron/

Trong bài này, tôi sẽ giới thiệu thuật toán đầu tiên trong Classification có tên là Perceptron Learning Algorithm (PLA) hoặc đôi khi được viết gọn là Perceptron.

Perceptron là một thuật toán Classification cho trường hợp đơn giản nhất: chỉ có hai class (lớp) (bài toán với chỉ hai class được gọi là binary classification) và cũng chỉ hoạt động được trong một trường hợp rất cụ thể. Tuy nhiên, nó là nền tảng cho một mảng lớn quan trọng của Machine Learning là Neural Networks và sau này là Deep Learning. (Tại sao lại gọi là Neural Networks - tức mạng dây thần kinh - các bạn sẽ được thấy ở cuối bài).

Giả sử chúng ta có hai tập hợp dữ liệu đã được gán nhãn được minh hoạ trong Hình 1 bên trái dưới đây. Hai class của chúng ta là tập các điểm màu xanh và tập các điểm màu đỏ. Bài toán đặt ra là: từ dữ liệu của hai tập được gán nhãn cho trước, hãy xây dựng một classifier (bộ phân lớp) để khi có một điểm dữ liệu hình tam giác màu xám mới, ta có thể dự đoán được màu (nhãn) của nó.

![Hình 0](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/pla1.png?raw=true)
![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/pla2.png?raw=true)

Hiểu theo một cách khác, chúng ta cần tìm lãnh thổ của mỗi class sao cho, với mỗi một điểm mới, ta chỉ cần xác định xem nó nằm vào lãnh thổ của class nào rồi quyết định nó thuộc class đó. Để tìm lãnh thổ của mỗi class, chúng ta cần đi tìm biên giới (boundary) giữa hai lãnh thổ này. Vậy bài toán classification có thể coi là bài toán đi tìm boundary giữa các class. Và boundary đơn giản nhât trong không gian hai chiều là một đường thằng, trong không gian ba chiều là một mặt phẳng, trong không gian nhiều chiều là một siêu mặt phẳng (hyperplane) (tôi gọi chung những boundary này là đường phẳng). Những boundary phẳng này được coi là đơn giản vì nó có thể biểu diễn dưới dạng toán học bằng một hàm số đơn giản có dạng tuyến tính, tức linear. Tất nhiên, chúng ta đang giả sử rằng tồn tại một đường phẳng để có thể phân định lãnh thổ của hai class. Hình 1 bên phải minh họa một đường thẳng phân chia hai class trong mặt phẳng. Phần có nền màu xanh được coi là lãnh thổ của lớp xanh, phần có nên màu đỏ được coi là lãnh thổ của lớp đỏ. Trong trường hợp này, điểm dữ liệu mới hình tam giác được phân vào class đỏ.

### Bài toán Perceptron : Cho hai class được gán nhãn, hãy tìm một đường phẳng sao cho toàn bộ các điểm thuộc class 1 nằm về 1 phía, toàn bộ các điểm thuộc class 2 nằm về phía còn lại của đường phẳng đó. Với giả định rằng tồn tại một đường phẳng như thế.

Nếu tồn tại một đường phẳng phân chia hai class thì ta gọi hai class đó là linearly separable. Các thuật toán classification tạo ra các boundary là các đường phẳng được gọi chung là Linear Classifier.

#### Mô tả bài toán:
![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/1.PNG?raw=true)
![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/pla4.png?raw=true)
![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/2.PNG?raw=true)
#### Xây dựng Loss Function

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/3.PNG?raw=true)
![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/4.PNG?raw=true)
![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/5.PNG?raw=true)

#### Có chỗ chứng minh hội tụ ý cơ mà mình chả hiểu gì cả mới khổ

#### Mô hình Neural Network đầu tiên

Má nó hay !!!!!!!!!!!
Sao người ta lại hình dung đc nó ra như thế nhỉ...LOLLLLLLLLLLLLL

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/6.PNG?raw=true)
![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/7.PNG?raw=true)
![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson7/img/8.PNG?raw=true)

## Thảo luận

- PLA có thể cho vô số nghiệm khác nhau
- PLA đòi hỏi dữ liệu linearly separable (tách rời)
### Pocket Algorithm
Một cách tự nhiên, nếu có một vài nhiễu, ta sẽ đi tìm một đường thẳng phân chia hai class sao cho có ít điểm bị misclassified nhất. Việc này có thể được thực hiện thông qua PLA với một chút thay đổi nhỏ như sau:

- Giới hạn số lượng vòng lặp của PLA.
- Mỗi lần cập nhật nghiệm w mới, ta đếm xem có bao nhiêu điểm bị misclassified. Nếu là lần đầu tiên, giữ lại nghiệm này trong pocket (túi quần). Nếu không, so sánh số điểm misclassified này với số điểm misclassified của nghiệm trong pocket, nếu nhỏ hơn thì lôi nghiệm cũ ra, đặt nghiệm mới này vào.
Thuật toán này giống với thuật toán tìm phần tử nhỏ nhất trong 1 mảng.