import tensorflow as tf

TFLITE_PATH = r"C:\Users\ThisPC\Documents\GitHub\Simple_GLKA\runs\dual_ccmt_augument\weights\tflite\best_deploy_int8.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

# Lấy chi tiết toàn bộ ops trong graph (cần dùng _get_ops_details, API nội bộ nhưng hoạt động tốt để debug)
try:
    ops_details = interpreter._get_ops_details()
except AttributeError:
    ops_details = None

print("=" * 80)
print("DANH SÁCH TẤT CẢ OP TRONG MODEL (lọc theo từ khóa 'pool'):")
print("=" * 80)

if ops_details:
    for op in ops_details:
        op_name = op.get('op_name', '')
        if 'pool' in op_name.lower():
            print(f"index={op['index']:>4}  op_name={op_name}")
else:
    print("Không lấy được _get_ops_details, in toàn bộ tensor thay thế:")

print()
print("=" * 80)
print("DANH SÁCH TẤT CẢ TENSOR (lọc theo từ khóa 'pool'):")
print("=" * 80)

tensor_details = interpreter.get_tensor_details()
for t in tensor_details:
    name = t['name']
    if 'pool' in name.lower():
        print(f"index={t['index']:>4}  name={name!r}  dtype={t['dtype']}  shape={t['shape']}")

print()
print("=" * 80)
print("TOÀN BỘ TENSOR (không lọc) - để đối chiếu nếu không thấy 'pool' ở trên:")
print("=" * 80)
for t in tensor_details:
    print(f"index={t['index']:>4}  name={t['name']!r}")