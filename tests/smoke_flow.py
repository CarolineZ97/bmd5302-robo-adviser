"""Manual smoke-test of the full conversation flow. Run with: python tests/smoke_flow.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from state_machine import SessionState, handle_user_input, greet

s = SessionState()
print("--- greet ---")
for seg in greet():
    print("[BOT]", seg.text[:100])

# Part-2 baseline answers: D,B,A,A,A,A,D,D,C,E -> total 48 -> R3 -> A=4
flows = ["start", "D", "B", "A", "A", "A", "A", "D", "D", "C", "E"]
for msg in flows:
    print(f"\n[USER] {msg}")
    for seg in handle_user_input(s, msg):
        print(f"[BOT/{seg.kind}]", seg.text[:150].replace("\n", " / "))

print("\n--- state after questionnaire ---")
print("phase=", s.phase.value, "conflicts so far=", s.unresolved_conflicts)

# Resolve remaining conflicts (if any)
guard = 0
while s.phase.value == "conflict_check" and guard < 5:
    print("\n[USER] A")
    for seg in handle_user_input(s, "A"):
        print(f"[BOT/{seg.kind}]", seg.text[:150].replace("\n", " / "))
    guard += 1

print("\n[USER] continue")
for seg in handle_user_input(s, "continue"):
    print(f"[BOT/{seg.kind}]", seg.text[:200].replace("\n", " / "))

print("\nweights non-zero:",
      sorted(((k, round(v * 100, 2)) for k, v in s.weights.items() if v > 0.005),
             key=lambda x: -x[1]))
print("A=", s.A_value, "data_source=", s.data_source)

print("\n[USER] what if A = 2 ?")
for seg in handle_user_input(s, "what if A = 2 ?"):
    print(f"[BOT/{seg.kind}]", seg.text[:200].replace("\n", " / "))
print("A after change=", s.A_value)
print("new weights:",
      sorted(((k, round(v * 100, 2)) for k, v in s.weights.items() if v > 0.005),
             key=lambda x: -x[1]))

print("\n[USER] explain sharpe")
for seg in handle_user_input(s, "explain sharpe"):
    print(f"[BOT/{seg.kind}]", seg.text[:200].replace("\n", " / "))

print("\n[USER] export pdf")
for seg in handle_user_input(s, "export pdf"):
    print(f"[BOT/{seg.kind}]", seg.text[:150])

print("\n[USER] restart")
for seg in handle_user_input(s, "restart"):
    print(f"[BOT/{seg.kind}]", seg.text[:150])
print("phase after restart=", s.phase.value, "current_q=", s.current_q)
