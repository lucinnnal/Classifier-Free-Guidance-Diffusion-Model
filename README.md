This is Conditional Diffusion model Code Scratch 
Training with MNIST data and give a label when sampling

How?: We add label info to Neural Net

# 확산 모델 : 주어진 샘플들에 유사한 샘플을 만들어 내는 모델 -> P(x)
# 조건부 확산 모델 : 조건을 넣어주어서 내가 원하는 샘플을 만들어 내는 모델 (5를 집어넣으면 -> 5라는 이미지가 나오도록) -> P(x|y) 
# 조건부 확산 모델의 y는 텍스트, 이미지, 레이블 등등이 될 수 있음

What I learned today..
(base) kipyokim@gimgipyoui-MacBookAir Conditional Diffusion % pwd
/Users/kipyokim/Desktop/Conditional Diffusion
(base) kipyokim@gimgipyoui-MacBookAir Conditional Diffusion % touch requirements.txt 
(base) kipyokim@gimgipyoui-MacBookAir Conditional Diffusion % touch .gitignore      -> git이 추적하지 않는 파일들을 모아놓은곳.. 예를들어, 가상환경 파일 등등
(base) kipyokim@gimgipyoui-MacBookAir Conditional Diffusion % touch README.md