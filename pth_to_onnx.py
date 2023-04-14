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
            model_storage_directory="easyocr_training/models/base"
            )
    return reader.recognizer, reader.detector, reader.character

def pth_to_onnx(model, input_size, file_name, output_names):
    model.eval()

    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names
    )

if __name__ == "__main__":
    rec_model, dec_model, character = get_models()
    code.interact(local=dict(globals(), **locals()))