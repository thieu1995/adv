
## Install new environment
```code
conda update -n base -c defaults conda

conda create -n advertiser python=3.7.3
source activate advertiser

conda install -c conda-forge pastescript
conda install -c conda-forge cherrypy
conda install -c conda-forge flask
conda install -c conda-forge pyspark
```

## Hướng dẫn chạy 
```code 
- Tên các file chạy nằm rõ ràng ở trong 2 list_files và list_files_type, chỉ việc copy thay thế vào cái FILE_USERS_NAME

- chỉ cần xem mình đang chạy file loại nào: Vd nếu chay file loại có chữ type ở trong tên file, như user_20_type 
- Thì phải set: FILE_ITEMS_NAME = "item_industry_types"

- Còn nếu tên file mà không có chữ type: vd: user_20 
- Thì phải set: FILE_ITEMS_NAME = "item_industry_types_and_hour"
```

## Code làm những gì 
```code 
- Đầu tiên nó load file users và file items vào spark 
- Tiếp đến chia file users thành train, validation, test 
- Tiếp đến huấn luyện với 3 tham số là: 
    + iteration: số vòng lặp 
    + rank : Số latent k trong cái matrix reduction 
    + regularization_parameters: Như kiểu cái regularization trong mạng nơ-ron ấy 
- Sau khi huấn luyện với toàn bộ các bộ tham số đó, nó sử dụng tập validation để tìm ra bộ tham số tốt nhất 
- Sau đó tính RMSE trên tập test dùng bộ tham số tốt nhất vừa tìm được ở bên trên 

- Giờ hệ gợi ý cho người dùng mới vào: 
- Có đoạn thêm 1 vài thông tin cho user id = 0 đó (Coi đây là user mới vào và có 1 số thông tin)
- Ta phải thêm các thông tin của user mới đó vào bộ dataset users chính, sau đó train lại 
- Train lại thì nhanh hơn rất nhiều (có hàm tính thời gian trong code đó)
- Đoạn gần cuối là đưa ra các items có kết quả tốt nhất, trong đó mỗi items phải có ít nhất tổng số lượt click trước đó là 20

- Cuối cùng thì ta có thể đự doán số lượt click của 1 user lên 1 advertiser bất kì.
 
- Dòng cuối cùng là save và load model đã train 
```




## Tutorials
```code 
https://towardsdatascience.com/large-scale-jobs-recommendation-engine-using-implicit-data-in-pyspark-ccf8df5d910e?fbclid=IwAR0HPG89mwaduNYhSfjxj88mvDFRWG5u6WgMgKshDmwkeywa3mZSFv4Ck1g
https://github.com/Akxay/recommendation_engine/blob/master/Jobs_RE_spark.ipynb


https://www.codementor.io/jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw?fbclid=IwAR0dXH0tMgkapGmlKgX_i2Ih-hlaCu11I2cymLv9obnSXKHhxvoHdfYFLw4

https://www.codementor.io/jadianes/building-a-web-service-with-apache-spark-flask-example-app-part2-du1083854

https://github.com/jadianes/spark-movie-lens
```



