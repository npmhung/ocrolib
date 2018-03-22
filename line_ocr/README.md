DEMO step
1. ssh vào server
ssh flax@65.49.54.195
password: FL@X!!!Cinnamon
2. activate toni's python environment
source toni_work/venv_p35/bin/activate
3. enter code folder
cd toni_work/flex_scan_engine2/flex_scan_engine/ocrolib/line_ocr/
4. run model
python train.py --dir_test='./data/japanese_name/test_data64kanjihira/' --test_file_out=./out.txt
Mọi người có thể thay folder dir_test bằng folder chứa ảnh 64xN, file ảnh tên là nội dung text trong ảnh. 
Sau khi chạy xong, predict string và real string sẽ được print theo thứ tự trái qua phải trong test_file_out, mỗi dòng tương ứng với 1 ảnh
Đã có sẵn 2 folders chứa ảnh test sing ngẫu nhiên: test_data64kanjihira và test_data64katakana
Trong quá trình chạy, có print ra màn hình mẫu 1 số cặp predict real. Cuối cùng là print statistics theo thứ tự từ trái qua phải: line accuracy, character accuracy. edit distance
5. api:
5.1 api load model: train.load_model
this api is called one time only, before prediction
5.2 api predict multiple files: train.predict_folder
when this api is called, it will predict all images in a specified folder, return list of predicted string corresponding
to ordered image files
5.3 api predict single file: train.predict_path
when this api is called, it will predict the image path, return one string
Example usages: see file model_api.py