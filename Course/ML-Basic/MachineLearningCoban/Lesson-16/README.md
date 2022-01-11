# Soft Margin Support Vector Machine

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/1.PNG?raw=true)

Có hai trường hợp dễ nhận thấy SVM làm việc không hiệu quả hoặc thậm chí không làm việc:

+ Trường hợp 1: Dữ liệu vẫn linearly separable như Hình 1a) nhưng có một điểm nhiễu của lớp tròn đỏ ở quá gần so với lớp vuông xanh. Trong trường hợp này, nếu ta sử dụng SVM thuần thì sẽ tạo ra một margin rất nhỏ. Ngoài ra, đường phân lớp nằm quá gần lớp vuông xanh và xa lớp tròn đỏ. Trong khi đó, nếu ta hy sinh điểm nhiễu này thì ta được một margin tốt hơn rất nhiều được mô tả bởi các đường nét đứt. SVM thuần vì vậy còn được coi là nhạy cảm với nhiễu (sensitive to noise).

+ Trường hợp 2: Dữ liệu không linearly separable nhưng gần linearly separable như Hình 1b). Trong trường hợp này, nếu ta sử dụng SVM thuần thì rõ ràng bài toán tối ưu là infeasible, tức feasible set là một tập rỗng, vì vậy bài toán tối ưu SVM trở nên vô nghiệm. Tuy nhiên, nếu ta lại chịu hy sinh một chút những điểm ở gần biên giữa hai classes, ta vẫn có thể tạo được một đường phân chia khá tốt như đường nét đứt đậm. Các đường support đường nét đứt mảnh vẫn giúp tạo được một margin lớn cho bộ phân lớp này. Với mỗi điểm nằm lần sang phía bên kia của các đường suport (hay đường margin, hoặc đường biên) tương ứng, ta gọi điểm đó rơi vào vùng không an toàn. Chú ý rằng vùng an toàn của hai classes là khác nhau, giao nhau ở phần nằm giữa hai đường support.

Trong cả hai trường hợp trên, margin tạo bởi đường phân chia và đường nét đứt mảnh còn được gọi là soft margin (biên mềm). Cũng theo cách gọi này, SVM thuần còn được gọi là Hard Margin SVM (SVM biên cứng).

Trong bài này, chúng ta sẽ tiếp tục tìm hiểu một biến thể của Hard Margin SVM có tên gọi là Soft Margin SVM.

Bài toán tối ưu cho Soft Margin SVM có hai cách tiếp cận khác nhau, cả hai đều mang lại những kết quả thú vị và có thể phát triển tiếp thành các thuật toán SVM phức tạp và hiệu quả hơn.

Cách giải quyết thứ nhất là giải một bài toán tối ưu có ràng buộc bằng cách giải bài toán đối ngẫu giống như Hard Margin SVM; cách giải dựa vào bài toán đối ngẫu này là cơ sở cho phương pháp Kernel SVM cho dữ liệu thực sự không linearly separable mà tôi sẽ đề cập trong bài tiếp theo. Hướng giải quyết này sẽ được tôi trình bày trong Mục 3 bên dưới.

Cách giải quyết thứ hai là đưa về một bài toán tối ưu không ràng buộc. Bài toán này có thể giải bằng các phương pháp Gradient Descent. Nhờ đó, cách giải quyết này có thể được áp dụng cho các bài toán large cale. Ngoài ra, trong cách giải này, chúng ta sẽ làm quen với một hàm mất mát mới có tên là hinge loss. Hàm mất mát này có thể mở rộng ra cho bài toán multi-class classification mà tôi sẽ đề cập sau 2 bài nữa (Multi-class SVM). Cách phát triển từ Soft Margin SVM thành Multi-class SVM có thể so sánh với cách phát triển từ Logistic Regression thành Softmax Regression. Hướng giải quyết này sẽ được tôi trình bày trong Mục 4 bên dưới.

Trước hết, chúng ta cùng đi phân tích bài toán.

## Phân tích toán học

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/2.PNG?raw=true)

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/3.PNG?raw=true)

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/4.PNG?raw=true)

## Bài toán đối ngẫu Lagrange

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/5.PNG?raw=true)

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/6.PNG?raw=true)

![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/7.PNG?raw=true)

![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/8.PNG?raw=true)

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/9.PNG?raw=true)

## Bài toán tối ưu không ràng buộc cho Soft Margin SVM

![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/10.PNG?raw=true)

![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/11.PNG?raw=true)

![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/12.PNG?raw=true)

![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/13.PNG?raw=true)

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/14.PNG?raw=true)

![Hình 15](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/15.PNG?raw=true)

![Hình 16](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/16.PNG?raw=true)

![Hình 17](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/17.PNG?raw=true)

![Hình 18](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/18.PNG?raw=true)

## Kiểm chứng bằng lập trình

read here: https://machinelearningcoban.com/2017/04/13/softmarginsmv/

source code: https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/20_softmarginsvm/plt/softmargin%20SVM%20Example.ipynb

![Hình 19](https://github.com/lacie-life/ML-basic/blob/master/Lesson-16/img/19.PNG?raw=true)

## Tóm tắt và thảo luận

+ SVM thuần (Hard Margin SVM) hoạt động không hiệu quả khi có nhiễu ở gần biên hoặc thậm chí khi dữ liệu giữa hai lớp gần linearly separable. Soft Margin SVM có thể giúp khắc phục điểm này.

+ Trong Soft Margin SVM, chúng ta chấp nhận lỗi xảy ra ở một vài điểm dữ liệu. Lỗi này được xác định bằng khoảng cách từ điểm đó tới đường biên tương ứng. Bài toán tối ưu sẽ tối thiểu lỗi này bằng cách sử dụng thêm các biến được gọi là slack varaibles.

+ Để giải bài toán tối ưu, có hai cách khác nhau. Mỗi cách có những ưu, nhược điểm riêng, các bạn sẽ thấy trong các bài tới.

+ Cách thứ nhất là giải bài toán đối ngẫu. Bài toán đối ngẫu của Soft Margin SVM rất giống với bài toán đối ngẫu của Hard Margin SVM, chỉ khác ở ràng buộc chặn trên của các nhân tử Laggrange. Ràng buộc này còn được gọi là box costraint.

+ Cách thứ hai là đưa bài toán về dạng không ràng buộc dựa trên một hàm mới gọi là hinge loss. Với cách này, hàm mất mát thu được là một hàm lồi và có thể giải được khá dễ dàng và hiệu quả bằng các phương pháp Gradient Descent.

+ Trong Soft Margin SVM, có một hằng số phải được chọn, đó là C. Hướng tiếp cận này còn được gọi là C-SVM. Ngoài ra, còn có một hướng tiếp cận khác cũng hay được sử dụng, gọi là ν-SVM,



