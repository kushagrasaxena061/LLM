import re

with open("train.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the 50-step silencer with a fast 10-step update
code = re.sub(
    r"if step % 50 == 0:\s*print\(f\"Step \{step\}.*?\)",
    "if step > 0 and step % 10 == 0:\n            print(f\"Step {step} | Loss: {loss.item():.4f}\")",
    code
)

with open("train.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ Terminal un-silenced! It will now print every 10 steps.")
