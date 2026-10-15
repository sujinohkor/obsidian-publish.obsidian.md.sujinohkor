# [[Enabling Uniform]]
	Enabling Uniform Computer Interaction Experience for Blind Users through Large Language Models
	(ASSETS '24) Satwik Ram Kodandaram
- 핵심: LLM을 이용해 시각장애인이 자연어로 컴퓨터를 조작하도록 하는 기술
- 문제: 시각장애인은 프로그램마다 다른 단축키와 스크린리더 조작법을 익혀야 함
- 방법: Savant라는 LLM 기반 보조기술을 개발. 사용자의 자연어 명령을 해석하고, 필요한 스크린리더 조작 과정을 자동으로 수행
- 결과: 시각장애인 11명을 대상으로 실험한 결과, 기존 방식보다 작업 효율성과 사용성이 유의미하게 개선
- 의의: 프로그램마다 조작법을 외울 필요 없이, 동일한 자연어 명령으로 여러 프로그램을 사용할 수 있도록 함

# [[EyeFormer]]
	EyeFormer: Predicting Personalized Scanpaths with Transformer-Guided Reinforcement Learning
	(UIST '24) Yue Jiang
- 핵심: Transformer와 강화학습으로 사용자 개인의 시선 이동 경로를 예측하는 기술
- 문제: 기존 시선 예측 모델은 평균적인 시선 패턴을 예측하는 데 집중하여 개인별 차이를 충분히 반영하지 못함
- 방법: Transformer와 심층 강화학습을 결합한 EyeFormer를 개발. 소수의 개인 시선 데이터만으로 시선의 이동 위치, 순서, 응시 시간을 예측
- 결과: 기존 모델 대비 여러 예측 지표에서 우수한 성능을 보였으며, 예측 결과를 활용해 사용자별 GUI 레이아웃을 최적화하는 응용 가능성을 보여줌
- 의의: 사용자 개인의 시각적 주의 특성을 반영해 인터페이스를 자동으로 개선할 수 있다는 점에서 의미가 있음

# [[Morae]]
	Morae: Proactively Pausing UI Agents for User Choices
	(UIST '25) Yi-Hao Peng
- 핵심: AI 에이전트가 중요한 선택 순간에 작업을 멈추고 사용자에게 결정권을 넘기는 기술
- 문제: 기존 AI 에이전트는 사용자의 세부적인 선호도를 확인하지 않고 작업을 자동으로 완료하는 경우가 있음
- 방법: Morae라는 AI 에이전트를 개발. 작업 중 선택이 필요한 상황을 감지하면 자동화를 일시 중지하고, 사용자에게 선택지를 제시
- 결과: 256개 작업에 대한 기술 평가와 시각장애인·저시력 사용자 10명의 실험에서 기존 에이전트 및 OpenAI Operator보다 높은 작업 성공률과 사용자 선호도에 부합하는 선택을 보였음
- 의의: AI 자동화와 인간의 의사결정권을 결합하는 Human-AI Interaction 연구

# [[Navigal]]
	 Navigal: Generating Adaptive Interfaces for Personalized Screen Reader Navigation
	(ACM '26) Anonymous
- 핵심: 사용자가 원하는 기능만 모아 개인화된 웹 인터페이스를 자동 생성하는 기술
- 문제: 시각장애인은 웹사이트마다 구조가 달라 원하는 버튼이나 기능을 찾기 위해 많은 탐색을 해야 함
- 방법: Navigal이라는 브라우저 확장 프로그램을 개발. 사용자가 자연어로 원하는 기능을 설명하면 해당 기능을 모은 인터페이스를 생성하고, 다른 웹페이지에서도 자동으로 연결
- 결과: 20개 웹 작업 중 18개를 지원했고, 기존 순차 탐색 대비 키 입력을 약 80% 줄였음. 시각장애인 7명 대상 평가에서도 탐색 부담과 인지적 부하가 감소
- 의의: AI가 모든 작업을 대신 수행하는 것이 아니라, 사용자가 직접 조작할 수 있는 맞춤형 인터페이스를 제공한다는 점에서 차별화

# [[Slide Gestalt]]
	 Slide Gestalt: Automatic Structure Extraction in Slide Decks for Non-Visual Access
	(CHI '23) Yi-Hao Peng
- 핵심: 파워포인트 슬라이드의 구조를 자동 분석하여 시각장애인이 쉽게 탐색하도록 하는 기술
- 문제: 스크린리더는 슬라이드에 나타난 시각적 계층구조를 제대로 전달하지 못함
- 방법: 슬라이드 간 텍스트와 시각적 유사성을 분석하여 장·절 등의 계층구조를 자동 추출하는 Slide Gestalt를 개발
- 결과: 시각장애인 및 저시력 사용자 8명의 평가에서, 슬라이드의 구조를 파악하고 필요한 내용으로 이동하는 데 도움이 되는 것으로 나타남
- 의의: 문서를 처음부터 끝까지 순서대로 읽는 대신, 목차와 계층구조를 이용해 원하는 부분으로 바로 이동할 수 있게 함

# [[The Dilemma]]
	- *The Dilemma of Building Do-It-Yourself (DIY) Solutions For Workplace Accessibility
	- *(CHI '25) Yoonha Cha
- 핵심: 시각장애인 개발자들이 직장에서 접근성 문제를 직접 해결하는 과정에 대한 연구
- 문제: 기존 개발도구의 접근성이 부족하여 시각장애인 개발자가 업무 수행에 어려움을 겪음
- 방법: 시각장애인 및 저시력 소프트웨어 전문가 30명을 인터뷰하여 직접 제작한 보조도구와 제작 동기를 조사
- 결과: 이들이 자신의 업무 문제를 해결하기 위해 도구를 직접 만들지만, 그 개발 자체가 추가적인 시간과 노동을 요구하는 Double Hacker Dilemma를 발견
- 의의: 접근성 문제를 장애인 개인의 노력으로 해결하도록 방치하기보다, 조직 차원의 지원과 보조도구 공유 체계가 필요하다는 점을 강조

# [[The Sky is the Limit]]
	The Sky is the Limit: Understanding How Generative AI can Enhance Screen Reader Users' Experience with Productivity Applications
	(CHI '25) Minoli Perera
- 핵심: 생성형 AI가 시각장애인의 문서 작업을 어떻게 도울 수 있는지 조사한 연구
- 문제: 시각장애인은 Word, Excel, PowerPoint 등의 생산성 프로그램을 사용하는 데 상당한 어려움을 겪음
- 방법: 스크린리더 사용자 99명을 설문조사하고, 16명을 추가 인터뷰하여 생성형 AI에 대한 요구사항과 인식을 분석
- 결과: 생성형 AI에 대한 기대가 높았지만, 기존 스크린리더 작업 방식과의 통합, 개인화, AI 작업 결과의 검증이 중요하다는 사실을 확인
- 의의: 시각장애인용 AI 보조기술을 설계할 때 어떤 기능과 사용자 통제 방식을 우선 고려해야 하는지 제시