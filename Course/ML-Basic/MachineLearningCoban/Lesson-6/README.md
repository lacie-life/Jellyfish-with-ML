## Gradient Descent

Tài liệu nè : https://machinelearningcoban.com/2017/01/12/gradientdescent/

Nhiều toán nè :)))

### thấy bảo đây là chân lý : luôn luôn đi ngược hướng với đạo hàm.
### Dùng để tối ưu việc tìm nghiệm cho các hàm mất mát (Loss function)

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/1.png?raw=true)

Local minimum : điểm cực tiểu
Global minimum : điểm mà tại đó hàm số đạt gái trị nhỏ nhất

Global minimum là một trường hợp đặc biệt của Local minimum

Tính chất: 

- Điểm local minimum x∗ của hàm số là điểm có đạo hàm f′(x∗) bằng 0. Hơn thế nữa, trong lân cận của nó, đạo hàm của các điểm phía bên trái x∗ là không dương, đạo hàm của các điểm phía bên phải x∗ là không âm.
- Đường tiếp tuyến với đồ thị hàm số đó tại 1 điểm bất kỳ có hệ số góc chính bằng đạo hàm của hàm số tại điểm đó.

Trong Machine Learning nói riêng và Toán Tối Ưu nói chung, chúng ta thường xuyên phải tìm giá trị nhỏ nhất (hoặc đôi khi là lớn nhất) của một hàm số nào đó. Ví dụ như các hàm mất mát trong hai bài Linear Regression và K-means Clustering. Nhìn chung, việc tìm global minimum của các hàm mất mát trong Machine Learning là rất phức tạp, thậm chí là bất khả thi. Thay vào đó, người ta thường cố gắng tìm các điểm local minimum, và ở một mức độ nào đó, coi đó là nghiệm cần tìm của bài toán.

Các điểm local minimum là nghiệm của phương trình đạo hàm bằng 0. Nếu bằng một cách nào đó có thể tìm được toàn bộ (hữu hạn) các điểm cực tiểu, ta chỉ cần thay từng điểm local minimum đó vào hàm số rồi tìm điểm làm cho hàm có giá trị nhỏ nhất (đoạn này nghe rất quen thuộc, đúng không?). Tuy nhiên, trong hầu hết các trường hợp, việc giải phương trình đạo hàm bằng 0 là bất khả thi. Nguyên nhân có thể đến từ sự phức tạp của dạng của đạo hàm, từ việc các điểm dữ liệu có số chiều lớn, hoặc từ việc có quá nhiều điểm dữ liệu.

Hướng tiếp cận phổ biến nhất là xuất phát từ một điểm mà chúng ta coi là gần với nghiệm của bài toán, sau đó dùng một phép toán lặp để tiến dần đến điểm cần tìm, tức đến khi đạo hàm gần với 0. Gradient Descent (viết gọn là GD) và các biến thể của nó là một trong những phương pháp được dùng nhiều nhất.

### Gradient Descent cho hàm 1 biến

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/2.PNG?raw=true)

Trong link tham khảo có VD hay vl ý. Xem đi :)))

### Gradient Descent cho hàm nhiều biến

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/3.PNG?raw=true)

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/4.PNG?raw=true)

Bài này đọc khá magic và 1 đống toán. Tạm skip

Đọc thêm tại link tài liệu ý

Hết phần 1 rồi !!!
\---------------------------------------------------------------------------------------------Phần 2 :)))))
Trong phần 1 của Gradient Descent (GD), tôi đã giới thiệu với bạn đọc về thuật toán Gradient Descent. Tôi xin nhắc lại rằng nghiệm cuối cùng của Gradient Descent phụ thuộc rất nhiều vào điểm khởi tạo và learning rate. Trong bài này, tôi xin đề cập một vài phương pháp thường được dùng để khắc phục những hạn chế của GD. Đồng thời, các thuật toán biến thể của GD thường được áp dụng trong các mô hình Deep Learning cũng sẽ được tổng hợp.

Tài liệu : https://machinelearningcoban.com/2017/01/16/gradientdescent2/

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/5.PNG?raw=true)

### Gradient Descent với Momentum 
    (trong tài liệu nha)
    Á đù. Cái này giống gia tốc và quán tính nhề. Hay phết :)))

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/6.PNG?raw=true)
![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/7.PNG?raw=true)
![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/M1.gif?raw=true)
![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/M2.gif?raw=true)

Hình bên trái là đường đi của nghiệm khi không sử dụng Momentum, thuật toán hội tụ sau chỉ 5 vòng lặp nhưng nghiệm tìm được là nghiệm local minimun.

Hình bên phải là đường đi của nghiệm khi có sử dụng Momentum, hòn bi đã có thể vượt dốc tới khu vực gần điểm global minimun, sau đó dao động xung quanh điểm này, giảm tốc rồi cuối cùng tới đích. Mặc dù mất nhiều vòng lặp hơn, GD với Momentum cho chúng ta nghiệm chính xác hơn. Quan sát đường đi của hòn bi trong trường hợp này, chúng ta thấy rằng điều này giống với vật lý hơn!

Ảo diệu đấy :))))

Cơ mà vì bạn này sử dụng đà nhiều quá nên lúc gần tới vị trí đích rồi mà vẫn mất kahs nhiều vòng lặp để hội tụ.

### Nesterov accelerated gradient (NAG)

Tài liệu (Lại toán) : https://cs231n.github.io/neural-networks-3/

#### Ý tưởng chính

![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/8.PNG?raw=true)
![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/9.PNG?raw=true)
![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/NAG.gif?raw=true)

Mình copy cả file notebook của ng ta đáp vào rùi đấy. Rảnh vô đọc chứ mình chả hiểu gì đâu :)))

Nao cũng thử nghiên cứu cái jupyternoetbook kia dùng sao làm 1 cái nhìn cho sướng :)))

Một vài thuật toán ngáo cần khác: https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent

Mình thấy toán là mình chạy rồi :)))) copy vô cho đẹp thôi

### Biến thể của GD

1. Batch Gradient Descent (Sử dung tất cả các điểm dữ liệu)
2. Stochastic Gradient Descent (SGD)
![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/10.PNG?raw=true)

Có 1 đoạn VD với Linear Regression trong tài liệu ý. Vô mà xem :))))

3. Mini-batch Gradient Descent

Có vẻ là link chân lý : https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson6/img/11.PNG?raw=true)

### Stopping Criteria (Điều kiện dừng)

Có một điểm cũng quan trọng mà từ đầu tôi chưa nhắc đến: khi nào thì chúng ta biết thuật toán đã hội tụ và dừng lại?

Trong thực nghiệm, có một vài phương pháp như dưới đây:

1. Giới hạn số vòng lặp: đây là phương pháp phổ biến nhất và cũng để đảm bảo rằng chương trình chạy không quá lâu. Tuy nhiên, một nhược điểm của cách làm này là có thể thuật toán dừng lại trước khi đủ gần với nghiệm.
2. So sánh gradient của nghiệm tại hai lần cập nhật liên tiếp, khi nào giá trị này đủ nhỏ thì dừng lại. Phương pháp này cũng có một nhược điểm lớn là việc tính đạo hàm đôi khi trở nên quá phức tạp (ví dụ như khi có quá nhiều dữ liệu), nếu áp dụng phương pháp này thì coi như ta không được lợi khi sử dụng SGD và mini-batch GD.
3. So sánh giá trị của hàm mất mát của nghiệm tại hai lần cập nhật liên tiếp, khi nào giá trị này đủ nhỏ thì dừng lại. Nhược điểm của phương pháp này là nếu tại một thời điểm, đồ thị hàm số có dạng bẳng phẳng tại một khu vực nhưng khu vực đó không chứa điểm local minimum (khu vực này thường được gọi là saddle points), thuật toán cũng dừng lại trước khi đạt giá trị mong muốn.
4. Trong SGD và mini-batch GD, cách thường dùng là so sánh nghiệm sau một vài lần cập nhật. Trong đoạn code Python phía trên về SGD, tôi áp dụng việc so sánh này mỗi khi nghiệm được cập nhật 10 lần. Việc làm này cũng tỏ ra khá hiệu quả.

### Một phương pháp tối ưu đơn giản khác: Newton's Method

Nhân tiện đang nói về tối ưu, tôi xin giới thiệu một phương pháp nữa có cách giải thích đơn giản: Newton’s method. Các phương pháp GD tôi đã trình bày còn được gọi là first-order methods, vì lời giải tìm được dựa trên đạo hàm bậc nhất của hàm số. Newton’s method là một second-order method, tức lời giải yêu cầu tính đến đạo hàm bậc hai.

Nhắc lại rằng, cho tới thời điểm này, chúng ta luôn giải phương trình đạo hàm của hàm mất mát bằng 0 để tìm các điểm local minimun. (Và trong nhiều trường hợp, coi nghiệm tìm được là nghiệm của bài toán tìm giá trị nhỏ nhất của hàm mất mát). Có một thuật toán nối tiếng giúp giải bài toán f(x)=0, thuật toán đó có tên là Newton’s method.

Biết thêm chi tiết thì đọc Link tham khảo chứ mình cũng chịu. Nhiều CT vl mà ngại gõ CT trong đây lắm :((((((((((((((






