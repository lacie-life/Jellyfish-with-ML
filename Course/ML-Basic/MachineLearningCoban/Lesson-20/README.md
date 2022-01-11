# Neighborhood-Based Collaborative Filtering

## Giới thiệu

Trong Content-based Recommendation Systems, chúng ta đã làm quen với một Hệ thống gợi ý sản phẩm đơn giản dựa trên đặc trưng của mỗi item. Đặc điểm của Content-based Recommendation Systems là việc xây dựng mô hình cho mỗi user không phụ thuộc vào các users khác mà phụ thuộc vào profile của mỗi items. Việc làm này có lợi thế là tiết kiệm bộ nhớ và thời gian tính toán. Đồng thời, hệ thống có khả năng tận dụng các thông tin đặc trưng của mỗi item như được mô tả trong bản mô tả (description) của mỗi item. Bản mô tả này có thể được xây dựng bởi nhà cung cấp hoặc được thu thập bằng cách yêu cầu users gắn tags cho items. Việc xây dựng feature vector cho mỗi item thường bao gồm các kỹ thuật Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP).

Cách làm trên có hai nhược điểm cơ bản. Thứ nhất, khi xây dựng mô hình cho một user, các hệ thống Content-based không tận dụng được thông tin từ các users khác. Những thông tin này thường rất hữu ích vì hành vi mua hàng của các users thường được nhóm thành một vài nhóm đơn giản; nếu biết hành vi mua hàng của một vài users trong nhóm, hệ thống nên suy luận ra hành vi của những users còn lại. Thứ hai, không phải lúc nào chúng ta cũng có bản mô tả cho mỗi item. Việc yêu cầu users gắn tags còn khó khăn hơn vì không phải ai cũng sẵn sàng làm việc đó; hoặc có làm nhưng sẽ mang xu hướng cá nhân. Các thuật toán NLP cũng phức tạp hơn ở việc phải xử lý các từ gần nghĩa, viết tắt, sai chính tả, hoặc được viết ở các ngôn ngữ khác nhau.

Những nhược điểm phía trên có thể được giải quyết bằng Collaborative Filtering (CF). Trong bài viết này, tôi sẽ trình bày tới các bạn một phương pháp CF có tên là Neighborhood-based Collaborative Filtering (NBCF). Bài tiếp theo sẽ trình bày về một phương pháp CF khác có tên Matrix Factorization Collaborative Filtering. Khi chỉ nói Collaborative Filtering, chúng ta sẽ ngầm hiểu rằng phương pháp được sử dụng là Neighborhood-based.

Ý tưởng cơ bản của NBCF là xác định mức độ quan tâm của một user tới một item dựa trên các users khác gần giống với user này. Việc gần giống nhau giữa các users có thể được xác định thông qua mức độ quan tâm của các users này tới các items khác mà hệ thống đã biết. Ví dụ, A, B đều thích phim Cảnh sát hình sự, tức đều rate bộ phim này 5 sao. Ta đã biết A cũng thích Người phán xử, vậy nhiều khả năng B cũng thích bộ phim này.

Các bạn có thể đã hình dung ra, hai câu hỏi quan trọng nhất trong một hệ thống Neighborhood-based Collaborative Filtering là:

Làm thế nào xác định được sự giống nhau giữa hai users?
Khi đã xác định được các users gần giống nhau (similar users) rồi, làm thế nào dự đoán được mức độ quan tâm của một user lên một item?
Việc xác định mức độ quan tâm của mỗi user tới một item dựa trên mức độ quan tâm của similar users tới item đó còn được gọi là User-user collaborative filtering. Có một hướng tiếp cận khác được cho là làm việc hiệu quả hơn là Item-item collaborative filtering. Trong hướng tiếp cận này, thay vì xác định user similarities, hệ thống sẽ xác định item similarities. Từ đó, hệ thống gợi ý những items gần giống với những items mà user có mức độ quan tâm cao.

Cấu trúc của bài viết như sau: Mục 2 sẽ trình bày User-user Collaborative Filtering. Mục 3 sẽ nêu một số hạn chế của User-user Collaborative Filtering và cách khắc phục bằng Item-item Collaborative Filtering. Kết quả của hai phương pháp này sẽ được trình bày qua ví dụ trên cơ sở dữ liệu MovieLens 100k trong Mục 4. Một vài thảo luận và Tài liệu tham khảo được cho trong Mục 5 và 6.

## User-user Collaborative Filtering

Read here: https://machinelearningcoban.com/2017/05/24/collaborativefiltering/