# Content Editing Task
python test.py \
--resume <please fill in your model checkpoint path> \
--transformer_config <please fill in your config file path> \
--image_dir example/0/i_s \
--target_txt example/0/i_t.txt \
--image_h 64

# Feature Permutation Task
python test_flexibility.py \
--resume <please fill in your model checkpoint path> \
--transformer_config <please fill in your config file path> \
--image_dir example/0/i_s \
--image_h 64