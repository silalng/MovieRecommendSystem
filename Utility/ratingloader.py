import csv
import redis

r = redis.Redis(host='sparkRecommend',port=6379,decode_responses=True)
with open("D:\\project_exercise\\MovieRecommendSystem\\Utility\\ratings.csv",encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = str("uid:" + row["uid"])
        movie = row["movie"]
        rating = row["rating"]
        r.lpush(key, str(movie+":"+rating))
        print(key, " ", f"{movie}:{rating}")
    print("导入完成")
