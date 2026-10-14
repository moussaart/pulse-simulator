# Product

## Register

product

## Users

PULSE researchers and operators run multi-agent localization and reinforcement-learning experiments. They use the training window alongside external RL clients to configure runs, monitor agent state, inspect localization accuracy and energy use, and intervene when an experiment needs to pause, reset, or stop.

## Product Purpose

PULSE provides a reliable environment for training and evaluating sensor-selection and localization policies against the simulator's real motion, UWB, IMU, and energy behavior. Success means operators can start a run confidently, understand its current state and quality, and keep the simulation progressing independently of GUI responsiveness.

## Brand Personality

Precise, instrument-like, and clear. The interface should communicate trustworthy measurements and explicit run state without hiding important controls or failure conditions.

## Anti-references

Avoid decorative or gamified dashboards, promotional/marketing layouts, ambiguous status indicators, and charts whose meaning depends on color alone. Do not let the viewer obscure whether the worker or external client is connected and progressing.

## Design Principles

- Make worker, client, and simulation state explicit.
- Put run control and failure recovery within easy reach.
- Present localization, reward, and energy as independently inspectable signals.
- Keep interaction usable by keyboard and make chart meaning robust to color-vision differences.
- Keep visualization responsive without making it responsible for simulation progress.

## Accessibility & Inclusion

Prioritize keyboard-operable controls, visible focus, readable labels, and explicit text for status and errors. Charts and agent identity must not rely on color alone; use distinct line styles, labels, or markers as well. Avoid rapid decorative motion.
