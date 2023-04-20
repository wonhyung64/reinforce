import openai


def chat(message, database):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": database},
            {"role": "user", "content": message},
        ],
    )
    print(response.choices[0]["message"]["content"])


if __name__ == "__main__":
    openai.api_key_path = "./cred/chatGPT_api_key.txt"

    database1 = "뮤즈는 서울대학교 경영대학 기타 동아리이며,\
        가입조건은 서울대학교 경영대학 재학생이어야 한다. \
        창립년도는 '1989년'이며,\
        창립멤버는 '87학번 박태성, 89학번 박종현 김태식, 김홍식, 손구호, 박운본입니다"
    message1 = "서울대학교 경영대학 기타 동아리인 '뮤즈'의 가입조건, 창립년도, 창립멤버에 대해서 알려줘."
    chat(message1, database1)

    reader = open('./data/name.csv', 'r', encoding="utf-8")
    table = reader.readlines()
    database2 = "".join(table)
    message2 = "표에서 middle name을 뽑아서 알려줘"
    chat(message2, database2)


    file_dir = "/Users/wonhyung64/data/trajectory/traj_05.csv"
    reader = open(file_dir, 'r', encoding="utf-8")
    table = reader.readlines()
    database3 = "".join(table)
    message3 = "표에서 sog 변수의 평균이랑 표준편차 알려줘"
    chat(message3, database3)
