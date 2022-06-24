rm -rf build
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

python -u quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log

python -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log

source compile.sh zcu104 ${BUILD} ${LOG}

xir png build/compiled_model/Actor_zcu104.xmodel model.png