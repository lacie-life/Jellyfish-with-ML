## K-means Clustering

Tài liệu nè : https://machinelearningcoban.com/2017/01/01/kmeans/

Thuật toán cơ bản trong Unsupervised learning đấy :)))))))
Trong thuật toán K-means clustering, chúng ta không biết nhãn (label) của từng điểm dữ liệu. Mục đích là làm thể nào để phân dữ liệu thành các cụm (cluster) khác nhau sao cho dữ liệu trong cùng một cụm có tính chất giống nhau.

Ý tưởng đơn giản nhất về cluster (cụm) là tập hợp các điểm ở gần nhau trong một không gian nào đó (không gian này có thể có rất nhiều chiều trong trường hợp thông tin về một điểm dữ liệu là rất lớn). Hình bên dưới là một ví dụ về 3 cụm dữ liệu (từ giờ tôi sẽ viết gọn là cluster)

![Hình 1](https://github.com/lacie-life/ML-basic/blob/master/Lesson4/img/figure_2.png?raw=true)

### Mục đích cuối cùng của thuật toán phân nhóm này là: từ dữ liệu đầu vào và số lượng nhóm chúng ta muốn tìm, hãy chỉ ra center của mỗi nhóm và phân các điểm dữ liệu vào các nhóm tương ứng. Giả sử thêm rằng mỗi điểm dữ liệu chỉ thuộc vào đúng một nhóm.

má nó magic học vl

center cuối cùng là trung bình cộng của các điểm trong cluster của nó...ghê vl

Tóm tắt thuật toán nè:
- Đầu vào: Dữ liệu X và số lượng cluster cần tìm K.
- Đầu ra: Các center M và label vector cho từng điểm dữ liệu Y.
1. Chọn K điểm bất kỳ làm các center ban đầu.
2. Phân mỗi điểm dữ liệu vào cluster có center gần nó nhất.
3. Nếu việc gán dữ liệu vào từng cluster ở bước 2 không thay đổi so với vòng lặp trước nó thì ta dừng thuật toán.
4. Cập nhật center cho từng cluster bằng cách lấy trung bình cộng của tất các các điểm dữ liệu đã được gán vào cluster đó sau bước 2.
5. Quay lại bước 2.

    Chúng ta có thể đảm bảo rằng thuật toán sẽ dừng lại sau một số hữu hạn vòng lặp. Thật vậy, vì hàm mất mát là một số dương và sau mỗi bước 2 hoặc 3, giá trị của hàm mất mát bị giảm đi. Theo kiến thức về dãy số trong chương trình cấp 3: nếu một dãy số giảm và bị chặn dưới thì nó hội tụ! Hơn nữa, số lượng cách phân nhóm cho toàn bộ dữ liệu là hữu hạn nên đến một lúc nào đó, hàm mất mát sẽ không thể thay đổi, và chúng ta có thể dừng thuật toán tại đây.

### Hạn chế
    - Chúng ta cần biết số lượng cluster cần clustering
    - Nghiệm cuối cùng phụ thuộc vào các centers được khởi tạo ban đầu
    - Các cluster cần có só lượng điểm gần bằng nhau
    - Các cluster cần có dạng hình tròn
    - Khi một cluster nằm phía trong 1 cluster khác
(Mấy cái này đọc link tài liệu để xem rõ hơn nha. Mà thực ra đọc luôn đi chứ đọc cái readme làm chi đm)

Example này chạy đi :))))

    https://machinelearningcoban.com/2017/01/04/kmeans2/

    

