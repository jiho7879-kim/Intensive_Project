import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 폰트 설정 함수
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Linux':
        # 리눅스 환경(서버)에서 많이 사용하는 폰트 지정
        # 만약 아래 폰트가 없다면, 터미널에서 'fc-list :lang=ko'로 
        # 서버에 설치된 한글 폰트 경로를 확인하세요.
        plt.rc('font', family='NanumGothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 코드 실행 전 설정 호출
set_korean_font()