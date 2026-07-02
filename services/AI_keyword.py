import os
import json
from google import genai
from google.genai import types 
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# 1. API 키 셋업
load_dotenv()
client = genai.Client(api_key=os.environ.get("API_KEY"))

# 2. 데이터 규격 정의
class AutoCaption(BaseModel):
    """자동 생성된 캡션 응답"""
    title: str
    script: str
    success: bool = True


def _build_image_parts(image_bytes_list: List[bytes], mime_types: Optional[List[str]] = None) -> List[types.Part]:
    """
    이미지 바이트 리스트를 Gemini에 넣을 Part 객체로 변환.
    mime_types가 주어지면 각 이미지의 실제 타입을 사용하고,
    없으면 안전하게 jpeg로 기본 처리.
    """
    parts = []
    for i, image_bytes in enumerate(image_bytes_list):
        mime_type = mime_types[i] if mime_types and i < len(mime_types) else "image/jpeg"
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    return parts


def _build_context_string(date: Optional[str] = None) -> str:
    """
    날짜 정보를 프롬프트에 넣을 문자열로 변환.
    날짜는 참고용일 뿐 AI 분석에 크게 영향을 주지는 않음.
    """
    if date:
        return f"[게시 예정 날짜]\n{date}\n"
    return ""


# ========================================================
# [메인 함수] 이미지를 받아서 바로 제목 + 캡션 1개 생성
# AI 방식: 작성자가 사진만 올리면 AI가 바로 제목+설명 작성
# ========================================================
def generate_caption_directly(
    image_bytes_list: List[bytes],
    mime_types: Optional[List[str]] = None,
    date: Optional[str] = None,
) -> AutoCaption:
    """
    이미지를 분석해서 제목과 상세 설명을 한 번에 생성합니다.
    
    입력: 이미지(들) + 날짜(선택)
    출력: {"title": "...", "script": "..."}
    
    사진을 한 번만 업로드하면 되고, "다시 생성" 버튼으로 다른 스타일을 얻을 수 있습니다.
    """
    contents_list = _build_image_parts(image_bytes_list, mime_types)
    context_string = _build_context_string(date)

    sys_instruct = f"""
너는 동아리 활동 사진을 보고, 그 활동을 기록하는 제목과 설명을 대신 써주는 AI야.
사진만 보고도 활동이 무엇이었는지 정확히 파악하고, 그 순간의 분위기를 담아 자연스럽게 기록해줘.

{context_string}

[작성 전, 사진을 먼저 이렇게 관찰해]
- 무슨 활동인지 (등산, 드라이브, 베이킹, 운동, 사진 촬영, 카페 모임, 봉사, 전시 관람, 캠핑, 낚시, 보드게임·방탈출, 스터디·독서모임, 공연 관람, 영화 감상, 악기 연주, 여행 등)
- 장소와 환경 (실내/실외, 자연/도심, 산·바다·강 같은 지형, 카페·스튜디오·강의실 같은 실내 공간, 계절감과 색감(단풍·벚꽃·눈), 날씨, 맑음/흐림, 시간대, 밝음/어두움/노을, 조명의 밝기나 분위기)
- 눈에 띄는 사물이나 디테일 (장비, 배경, 음식이나 음료, 풍경, 인원 수와 구성, 소품(피크닉 매트·텐트·카메라·악기 등), 사람들의 표정이나 손동작)
- 전체적인 분위기 (활기참, 차분함, 집중, 여유, 왁자지껄함, 고요함, 나른함, 몰입, 설렘 등)
관찰한 사실을 근거로 제목과 설명을 써야 해. 사진에 없는 건 절대 지어내지 마.

[제목 작성 규칙]
- 8~15자, 활동의 핵심이나 그 순간의 분위기를 한 줄로
- 사진에서 실제로 파악한 활동/장소가 드러나면 좋음
- 이모지나 특수기호는 쓰지 말 것

[설명 작성 규칙]
- 4~6문장, 일기를 쓰듯 차분하고 따뜻한 문체
- 사진에서 관찰한 활동, 장소, 분위기를 구체적으로 담기
- 활동의 흐름(무엇을 했고, 어땠고, 어떤 기분이었는지)이 자연스럽게 이어지게
- 담담하게, 그때의 기분이 억지스럽지 않게 묻어나게

[좋은 예시 - 이 패턴과 분량을 참고해]
(사진: 한강에서 자전거, 맑은 날)
제목: "한강 자전거 라이딩"
설명: "맑은 날 한강을 따라 자전거를 탔다. 물 위로 햇빛이 반사되는 걸 보면서 계속 페달을 밟았다. 중간에 잠깐 멈춰서 다리 위에서 강을 내려다봤는데 바람이 시원했다. 생각보다 오래 달려서 다리는 좀 뻐근했지만 기분은 가벼웠다. 다음에는 더 멀리까지 가보고 싶다."

(사진: 실내 스튜디오 조명 세팅)
제목: "조명 세팅하는 날"
설명: "스튜디오에서 인물 사진용 조명을 세팅했다. 각도를 바꿔가며 여러 번 테스트했는데 원하는 느낌이 쉽게 안 나왔다. 다 같이 모니터를 보면서 이게 낫다 저게 낫다 의견을 주고받았다. 생각보다 시간이 오래 걸렸지만 하나씩 맞춰가는 과정이 나름 재밌었다. 다음엔 좀 더 빨리 잡을 수 있을 것 같다."

(사진: 밤에 도심에서 러닝, 야경)
제목: "밤 러닝 한 바퀴"
설명: "저녁을 먹고 다 같이 러닝을 나갔다. 낮보다 선선해서 뛰기에 딱 좋은 날씨였다. 불 켜진 건물들을 지나면서 한 바퀴 도는데 생각보다 페이스가 잘 붙었다. 중간부터 숨이 차서 말수가 줄었지만 끝까지 다 같이 완주했다. 땀 흘리고 나니 머리가 개운해진 느낌이었다."

(사진: 차 안에서 노을 보이는 드라이브)
제목: "노을 보러 드라이브"
설명: "특별한 목적지 없이 그냥 차를 타고 나왔다. 창밖으로 노을이 지기 시작하는 걸 보면서 음악을 틀어놨다. 신호에 걸릴 때마다 하늘 색이 조금씩 바뀌는 게 보였다. 딱히 대단한 걸 한 건 아닌데 이런 시간이 오히려 오래 기억에 남을 것 같다."

(사진: 실내에서 보드게임 하는 모습, 밤)
제목: "밤새 보드게임"
설명: "다 같이 모여서 보드게임을 했다. 규칙을 설명하는 데만 한참 걸렸는데 막상 시작하니 다들 승부욕이 올라왔다. 이기려고 머리 굴리다가 어이없는 판단으로 지는 사람이 나올 때마다 웃음이 터졌다. 시간 가는 줄 모르고 몇 판을 연달아 했다. 별거 아닌데 이런 날이 제일 재밌는 것 같다."

(사진: 베이킹, 반죽과 오븐)
제목: "같이 만든 쿠키"
설명: "다 같이 쿠키를 만들어봤다. 반죽을 나눠서 모양을 내는데 각자 손재주가 티가 났다. 오븐에서 익는 냄새가 퍼지기 시작하니까 다들 앞에 모여서 구경했다. 몇 개는 좀 타긴 했지만 그것도 나름 맛있었다. 만드는 내내 손은 엉망이 됐어도 계속 웃으면서 했다."

[절대 금지]
- "최고예요", "정말 멋져요!", "이렇게 예쁜 곳은 처음" 같은 과장·감탄
- "빛나는", "소중한", "몽글몽글" 같은 뻔한 감성 단어
- 사진에 보이지 않는 정보를 추측해서 추가하는 것
- 광고나 홍보처럼 들리는 톤
"""

    contents_list.append("이 사진을 보고 제목과 상세 설명을 작성해줘. 자연스럽고 사실적으로.")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents_list,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "8~15자 내외의 활동 제목"
                    },
                    "script": {
                        "type": "string",
                        "description": "4~6문장의 활동 기록 설명"
                    }
                },
                "required": ["title", "script"]
            },
            temperature=1.1  # 매번 다른 스타일로 생성하기 위해 약간의 창의성 확보
        )
    )
    
    result = json.loads(response.text)
    return AutoCaption(
        title=result.get("title", ""),
        script=result.get("script", ""),
        success=True
    )


# ========================================================
# [로컬 테스트]
# ========================================================
if __name__ == "__main__":
    # 테스트할 이미지 파일명을 여기에 나열 (1장이든 여러 장이든 자유롭게)
    # 예) 단일: ["cat1.jpg"]   /   다중: ["cat1.jpg", "cat2.jpg", "cat3.jpg", "cat4.jpg"]
    target_images = ["cat1.jpg", "cat2.jpg", "cat3.jpg", "cat4.jpg"]

    print("\n" + "="*50)
    print(f"AI 캡션 생성 테스트 (이미지 {len(target_images)}장)")
    print("="*50 + "\n")

    image_bytes_list = []
    for path in target_images:
        if not os.path.exists(path):
            print(f"에러: '{path}' 파일이 없습니다!")
            exit()
        with open(path, "rb") as f:
            image_bytes_list.append(f.read())

    print("AI가 제목과 설명을 작성 중입니다...\n")
    caption = generate_caption_directly(image_bytes_list, date="2026-07-01")

    print("[생성된 캡션]")
    print("=" * 50)
    print(f"제목: {caption.title}")
    print(f"\n상세 설명:\n{caption.script}")
    print("=" * 50)

    # 다시 생성 테스트
    print("\n\n[다시 생성 - 다른 스타일]")
    caption2 = generate_caption_directly(image_bytes_list, date="2026-07-01")
    print("=" * 50)
    print(f"제목: {caption2.title}")
    print(f"\n상세 설명:\n{caption2.script}")
    print("=" * 50)