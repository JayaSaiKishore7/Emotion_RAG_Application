import traceback
try:
    import core.models as m
    print("OK: imported core.models")
    names = sorted(n for n in dir(m) if not n.startswith("_"))
    print("Exported names from core.models:")
    for n in names:
        print(" -", n)
except Exception as e:
    print("IMPORT ERROR:", e)
    traceback.print_exc()
###
