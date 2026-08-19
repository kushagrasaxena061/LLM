import re

with open("train.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the "print every step" logic with a 50-step silencer
code = re.sub(
    r"scheduler\.step\(\)\n\s*print\(f\"Step \{step\} \| Loss: \{loss\.item\(\):\.4f\}\"\)",
    "scheduler.step()\n        if step % 50 == 0:\n            print(f\"Step {step} | Loss: {loss.item():.4f}\")",
    code
)

with open("train.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ Print bottleneck removed! It will now print every 50 steps.")
