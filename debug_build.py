import dis
import inspect
import sys
import PyInstaller.__main__

original_get_const_info = dis._get_const_info

def patched_get_const_info(arg, constants):
    try:
        return original_get_const_info(arg, constants)
    except IndexError:
        with open("dis_error.log", "w", encoding="utf-8") as f:
            f.write(f"!!! DIS ERROR !!! Constants: {constants}, Arg: {arg}\n")
            # Try to find the calling frame that has `code_object`
            for frame_info in inspect.stack():
                if 'code_object' in frame_info.frame.f_locals:
                    co = frame_info.frame.f_locals['code_object']
                    f.write(f"Code object filename: {co.co_filename}\n")
                    f.write(f"Code object name: {co.co_name}\n")
                    break
        raise

dis._get_const_info = patched_get_const_info

if __name__ == '__main__':
    PyInstaller.__main__.run(["main.spec", "--clean", "--noconfirm"])
