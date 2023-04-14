import torch
import torch.onnx
import easyocr
from easyocr.craft import CRAFT
import code

def get_models():
    reader = easyocr.Reader(
            ['en'],
            gpu=False, 
            download_enabled=False, 
            model_storage_directory="models/base"
            )
    return reader.recognizer, reader.detector, reader.character

def pth_to_onnx(
        model, input_size, file_name, output_names,):
    model.eval()

    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )

if __name__ == "__main__":
    rec_model, dec_model, character = get_models()
    code.interact(local=dict(globals(), **locals()))
    # pth_to_onnx(dec_model, (3, 224, 224), "dec_test.onnx", ["boxs"])
    # pth_to_onnx(rec_model, (1, 224, 224), "rec_test.onnx", list(character))