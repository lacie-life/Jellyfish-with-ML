## Support Vector Machine

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/1.PNG?raw=true)

![Hình 2](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/2.PNG?raw=true)

![Hình 3](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/3.PNG?raw=true)

![Hình 4](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/4.PNG?raw=true)

## Xây dựng bài toán tối ưu cho SVM

![Hình 5](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/5.PNG?raw=true)

![Hình 6](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/6.PNG?raw=true)

![Hình 7](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/7.PNG?raw=true)

![Hình 8](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/8.PNG?raw=true)

![Hình 9](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/9.PNG?raw=true)

## Bài toán đối ngẫu cho SVM

![Hình 10](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/10.PNG?raw=true)

![Hình 11](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/11.PNG?raw=true)

![Hình 12](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/12.PNG?raw=true)

![Hình 13](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/13.PNG?raw=true)

![Hình 14](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/14.PNG?raw=true)

![Hình 15](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/15.PNG?raw=true)

![Hình 16](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/16.PNG?raw=true)

![Hình 17](https://github.com/lacie-life/ML-basic/blob/master/Lesson15/img/17.PNG?raw=true)

## Lập trình tìm nghiệm cho SVM

read here: https://machinelearningcoban.com/2017/04/09/smv/

## Tóm tắt và thảo luận

+ Với bài toán binary classification mà 2 classes là linearly separable, có vô số các siêu mặt phẳng giúp phân biệt hai classes, tức mặt phân cách. Với mỗi mặt phân cách, ta có một classifier. Khoảng cách gần nhất từ 1 điểm dữ liệu tới mặt phân cách ấy được gọi là margin của classifier đó.

+ Support Vector Machine là bài toán đi tìm mặt phân cách sao cho margin tìm được là lớn nhất, đồng nghĩa với việc các điểm dữ liệu an toàn nhất so với mặt phân cách.

+ Bài toán tối ưu trong SVM là một bài toán lồi với hàm mục tiêu là stricly convex, nghiệm của bài toán này là duy nhất. Hơn nữa, bài toán tối ưu đó là một Quadratic Programming (QP).

+ Mặc dù có thể trực tiếp giải SVM qua bài toán tối ưu gốc này, thông thường người ta thường giải bài toán đối ngẫu. Bài toán đối ngẫu cũng là một QP nhưng nghiệm là sparse nên có những phương pháp giải hiệu quả hơn.

+ Với các bài toán mà dữ liệu gần linearly separable hoặc nonlinear separable, có những cải tiền khác của SVM để thích nghi với dữ liệu đó. Mời bạn đón đọc bài tiếp theo.



