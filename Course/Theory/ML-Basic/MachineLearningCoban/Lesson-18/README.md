# Multi-class Support Vector Machine

## Từ Binary classification tới multi-class classification

Các phương pháp Support Vector Machine đã đề cập (Hard Margin, Soft Margin, Kernel) đều được xây dựng nhằm giải quyết bài toán Binary Classification, tức bài toán phân lớp với chỉ hai classes. Việc này cũng tương tự như Percetron Learning Algorithm hay Logistic Regression vậy. Các mô hình làm việc với bài toán có 2 classes còn được gọi là Binary classifiers. Một cách tự nhiên để mở rộng các mô hình này áp dụng cho các bài toán multi-class classification, tức có nhiều classes dữ liệu khác nhau, là sử dụng nhiều binary classifiers và các kỹ thuật như one-vs-one hoặc one-vs-rest. Cách làm này có những hạn chế như đã trình bày trong bài Softmax Regression.

## Mô hình end-to-end

Softmax Regression là mở rộng của Logistic Regression cho bài toán multi-class classification, có thể được coi là một layer của Neural Networks. Nhờ đó, Softmax Regression thường đươc sử dụng rất nhiều trong các bộ phân lớp hiện nay. Các bộ phân lớp cho kết quả cao nhất thường là một Neural Network với rất nhiều layers và layer cuối là một softmax regression, đặc biệt là các Convolutional Neural Networks. Các layer trước thường là kết hợp của các Convolutional layers và các nonlinear activation functions và pooling, các bạn tạm thời chưa cần quan tâm đến các layers phía trước này, tôi sẽ giới thiệu khi có dịp. Có thể coi các layer trước layer cuối là một công cụ giúp trích chọn đặc trưng của dữ liệu (Feature extraction), layer cuối là softmax regression, là một bộ phân lớp tuyến tính đơn giản nhưng rất hiệu quả. Bằng cách này, ta có thể coi là nhiều one-vs-rest classifers được huấn luyện cùng nhau, hỗ trợ lẫn nhau, vì vậy, một cách tự nhiên, sẽ có thể tốt hơn là huấn luyện từng classifier riêng lẻ.

Sự hiệu quả của Softmax Regression nói riêng và Convolutional Neural Networks nói chung là cả bộ trích chọn đặc trưng (feature extractor) và bộ phân lớp (classifier) được huấn luyện đồng thời. Điều này nghĩa là hai bộ phận này bổ trợ cho nhau trong quá trình huấn luyện. Classifier giúp tìm ra các hệ số hợp lý phù hợp với feature vector tìm được, ngược lại, feature extractor lại điều chỉnh các hệ số của các convolutional layer sao cho feature thu được là tuyến tính, phù hợp với classifier ở layer cuối cùng.

Tôi viết đến đây không phải là để giới thiệu về Softmax Regression, mà là đang nói chung đến các mô hình phân lớp hiện đại. Đặc điểm chung của chúng là feature extractor và classifier được huấn luyện một cách đồng thời. Những mô hình như thế này còn được gọi là end-to-end. Cùng xem lại mô hình chung cho các bài toán Machine Learning mà tôi đã đề cập trong Bài 11:

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/ML_models.png?raw=true)

Trong Hình 1, phần TRAINING PHASE, chúng ta có thể thấy rằng có hai khối chính là Feature Extraction và Classification/Regression/Clustering… Các phương pháp truyền thống thường xây dựng hai khối này qua các bước riêng rẽ. Phần Feature Extraction với dữ liệu ảnh có thể dùng các feature descriptor như SIFT, SURF, HOG; với dữ liệu văn bản thì có thể là Bag of Words hoặc TF-IDF. Nếu là các bài toán classification, phần còn lại có thể là SVM thông thường hay các bộ phân lớp truyền thống khác.

Với sự phát triển của Deep Learning trong những năm gần đây, người ta cho rằng các hệ thống end-to-end (từ đầu đến cuối) mang lại kết quả tốt hơn nhờ và việc các hai khối phía trên được huấn luyện cùng nhau, bổ trợ lẫn nhau. Thực tế cho thấy, các phương pháp state-of-the-art thường là các mô hình end-to-end.

Các phương pháp Support Vector Machine được chứng minh là tốt hơn Logistic Regression vì chúng có quan tâm đến việc tạo margin lớn nhất giữa các classes. Câu hỏi đặt ra là:

Liệu có cách nào giúp kết hợp SVM với Neural Networks để tạo ra một bộ phân lớp tốt với bài toán multi-class classification? Hơn nữa, toàn bộ hệ thống có thể được huấn luyện theo kiểu end-to-end?

Câu trả lời sẽ được tìm thấy trong bài viết này, bằng một phương pháp được gọi là Multi-class Support Vector Machine.

Và để cho bài viết hấp dẫn hơn, tôi xin giới thiệu luôn, ở phần cuối, chúng ta sẽ cùng lập trình từ đầu đến cuối để giải quyết bài toán phân lớp với bộ cơ sở dữ liệu nổi tiếng: CIFAR10.

### Bộ cơ sở dữ liệu CIFAR10

Bộ cơ sở dữ liệu CIFAR10 gồm 51000 ảnh khác nhau thuộc 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, và truck. Mỗi bức ảnh có kích thước 32×32 pixel. Một vài ví dụ cho mỗi class được cho trong Hình 2 dưới đây. 50000 ảnh được sử dụng cho training, 1000 ảnh còn lại được dùng cho test. Trong số 50000 ảnh training, 1000 ảnh sẽ được lấy ra ngẫu nghiên để làm validation set.

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/cifar.png?raw=true)

Đây là một bộ cơ sở dữ liệu tương đối khó vì ảnh nhỏ và object trong cùng một class cũng biến đổi rất nhiều về màu sắc, hình dáng, kích thước. Thuật toán tốt nhất hiện nay cho bài toán này đã đạt được độ chính xác trên 90%, sử dụng một Convolutional Neural Network nhiều lớp kết hợp với Softmax regression ở layer cuối cùng. Trong bài này, chúng ta sẽ sử dụng một mô hình neural network đơn giản không có hidden layer nào để giải quyết, kết quả đạt được là khoảng 40%, nhưng cũng là đã rất ấn tượng. Layer cuối là một layer Multi-class SVM. Tôi sẽ hướng dẫn các bạn lập trình cho mô hình này từ đầu đến cuối mà không sử dụng một thư viện đặc biệt nào ngoài numpy.

Bài toán này cũng như nội dung chính của bài viết được lấy từ Lecture notes: Linear Classifier II và Assignment #1 trong khoá học CS231n: Convolutional Neural Networks for Visual Recognition kỳ Winter 2016 của Stanford.

Trước khi đi vào mục xây dựng hàm mất mát cho Multi-class SVM, tôi muốn nhắc lại một chút về một chút feature engineering cho ảnh trong CIFAR-10 và bias trick nói chung trong Neural Networks.

### Image data preprocessing

Để cho mọi thứ được đơn giản và có được một mô hình hoàn chỉnh, chúng ta sẽ sử dụng phương pháp feature engineering đơn giản nhất: lấy trực tiếp tất cả các pixel trong mỗi ảnh và thêm một chút normalization.

+ Mỗi ảnh của CIFAR-10 đã có kích thước giống nhau 32×32 pixel, vì vậy việc đầu tiên chúng ta cần làm là kéo dài mỗi trong ba channels Red, Green, Blue của bức ảnh ra thành một vector có kích thước là 3×32×32=3072.
+ Vì mỗi pixel có giá trị là một số tự nhiên từ 0 đến 255 nên chúng ta cần một chút chuẩn hóa dữ liệu. Trong Machine Learning, một cách đơn giản nhất để chuẩn hóa dữ liệu là center data, tức làm cho mỗi feature có trung bình cộng bằng 0. Một cách đơn giản để làm việc này là ta tính trung bình cộng của tất cả các ảnh trong tập training để được ảnh trung bình, sau đó trừ từ tất cả các ảnh đi ảnh trung bình này. Tương tự, ta cũng dùng ảnh trung bình này để chuẩn hoá dữ liệu trong validation set và test set.

### Bias trick

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/1.PNG?raw=true)

## Xây dựng hàm mất mát cho Multi-class Support Vector Machine

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/2.PNG?raw=true)

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/3.PNG?raw=true)

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/4.PNG?raw=true)

![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/5.PNG?raw=true)

![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/6.PNG?raw=true)

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/7.PNG?raw=true)

![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/8.PNG?raw=true)

## Tính toán hàm mất mát và đạo hàm của nó

Để tối ưu hàm mất mát, chúng ta sử dụng phương pháp Stochastic Gradient Method. Điều này có nghĩa là chúng ta cần tính gradient tại mỗi vòng lặp. Đồng thời, loss sau mỗi vòng lặp cũng cần được tính để kiểm tra liệu thuật toán có hoạt động như ý muốn hay không.

Việc tính toán loss và gradient này không những cần phải chính xác mà còn cần được thực hiện càng nhanh càng tốt. Trong khi việc tính loss thường dễ thực hiện, việc tính gradient cần phải được kiểm tra kỹ càng hơn.

Để đảm bảo rằng loss và gradient được tính một cách chính xác và nhanh, chúng ta sẽ làm từng bước một. Bước thứ nhất là đảm bảo rằng các tính toán là chính xác, dù cách tính có rất chậm. Bước thứ hai là phải đảm bảo có cách tính hiệu quả để thuật toán chạy nhanh hơn. Hai bước này cần được thực hiện trên một lượng dữ liệu nhỏ để đảm bảo chúng được tính chính xác trước khi áp dụng thuật toán vào dữ liệu thật, thường có số điểm dữ liệu lớn và mỗi điểm dữ liệu cũng có số chiều lớn.

Hai mục nhỏ tiếp theo sẽ mô tả hai bước đã nêu ở trên.

![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/9.PNG?raw=true)

![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/10.PNG?raw=true)

![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/11.PNG?raw=true)

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/12.PNG?raw=true)

![Hình 15](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/13.PNG?raw=true)

### Cái này nó magic vl này : Tính hàm mất mát và đạo hàm của nó bằng cách vectorized

![Hình 16](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/14.PNG?raw=true)

![Hình 17](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/15.PNG?raw=true)

![Hình 18](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/16.PNG?raw=true)

![Hình 19](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/17.PNG?raw=true)

### Gradient Descent cho Multi-class SVM

![Hình 20](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/18.PNG?raw=true)

![Hình 21](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/19.PNG?raw=true)

![Hình 22](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/20.PNG?raw=true)

## Thảo luận

![Hình 23](https://github.com/lacie-life/ML-basic/blob/master/Lesson-18/img/21.PNG?raw=true)

Tài liệu tham khảo: 
https://machinelearningcoban.com/2017/04/28/multiclasssmv/
https://cs231n.github.io/linear-classify/
https://en.wikipedia.org/wiki/Hinge_loss