import openai


def chat(message):
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

    database = "뮤즈는 서울대학교 경영대학 기타 동아리이며,\
        가입조건은 서울대학교 경영대학 재학생이어야 한다. \
        창립년도는 '1989년'이며,\
        창립멤버는 '87학번 박태성, 89학번 박종현 김태식, 김홍식, 손구호, 박운본입니다"

    message = "서울대학교 경영대학 기타 동아리인 '뮤즈'의 가입조건, 창립년도, 창립멤버에 대해서 알려줘."

    chat(message)