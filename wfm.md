
# Waveform Synthesis - Derivations
### Narrator: SK

In this document we describe how waveforms move atoms from one trap to another. We first briefly describe how static waveforms are synthesizes; afterwards, we describe how _transition waveforms_ can be synthesized. Finally, we provide explicit forms of specific examples of transition waveforms.

**Transition waveforms** are waveforms that, over some duration $T$, transition the waveform amplitude, frequency, and overall phase. Examples of these waveforms are extraction waveforms (dynamic traps picking up atoms from a static trap), implantation waveforms (dynamic traps planting the atoms into a static trap), and displacement waveforms (dynamic traps moving a trap from one site to another).

## Static Waveforms

To each trap is associated a waveform responsible for holding the atom trapped in its place. The waveforms are defined by three variables: $(\alpha, \nu, \phi)$, where $\alpha$ is the amplitude of the static waveform associated with the trap, $\nu$ is the associated frequency, and $\phi$ is the overall phase.

The static waveform is thus defined as:

$$ f(t) = \alpha \sin(2\pi \nu t + \phi) $$

## Transition Waveforms

Each transition waveform is responsible for:
1) Transitioning the amplitude of the waveform
2) Transitioning the frequency of the waveform (chirping the frequency)
3) Transitioning the overall phase of the waveform

In otherwords, the transition waveform is a waveform that _smoothly_ transitions from a waveform $(\alpha_1, \nu_1, \phi_1)$ to $(\alpha_2, \nu_2, \phi_2)$. We will describe mathematically what _smooth_ means in this context.

A general transition waveform is thus of the following format:

$$ f(t) = \alpha(t) \sin(\phi(t)) , 0 \leq t \leq T $$

The transition waveform must satisfy the following conditions.

1) The function $\alpha(t)$ satisfies:
$$ \alpha(0) = \alpha_1, \alpha( T)=\alpha_2$$
2) The derivate of the $\alpha(t)$ function satisfies:
$$ \alpha'(0) = 0, \alpha'( T)=0$$
3) The function $\nu(t)$ satisfies ($k \in \mathbb{Z}$ is arbitrary):
$$ \phi(0) = \phi_1, \phi( T)=2\pi k + \phi_2$$
4) The derivate of the $\phi(t)$ function satisfies:
$$ \phi'(0) = 2\pi \nu_0, \phi'(T)=2\pi \nu_1$$
5) The second derivate of the $\phi(t)$ function satisfies:
$$ \phi''(0) = 0, \phi''(T)=0$$

All the equality signs above can be relaxed by allowing some error tolerance. Some waveforms, by construct, cannot satisfy all the conditions above simultaneously. Therefore, relaxing the conditions above is sometimes a necessity.

## Tanh Transition Waveforms

**Parameters**
$$ v_{\text{max}}\text{: Maximum velocity of the atom/Interatomic Distance} $$

**Waveform Description**

$$ \alpha(t) = \dfrac{\alpha_1 + \alpha_2}{2} + \dfrac{\alpha_2 - \alpha_1}{2} \tanh\left(2v_{\text{max}} \left[t - \dfrac{T}{2}\right]\right) $$

$$ \nu(t) = \dfrac{\nu_1 + \nu_2}{2} + \dfrac{\nu_2 - \nu_1}{2} \tanh\left(2v_{\text{max}} \left[t - \dfrac{T}{2}\right]\right) $$

**Equations**

$$ f(t) = \alpha(t) \sin\left(\phi_1 + 2\pi \tilde{\phi}(t) + \dfrac{\Delta \phi \cdot t}{T}\right) $$

Where:

$$ \alpha(t) = \dfrac{\alpha_1 + \alpha_2}{2} + \dfrac{\alpha_2 - \alpha_1}{2} \tanh\left(2v_{\text{max}}\left[t-\dfrac{T}{2}\right]\right) $$

$$ \tilde{\phi}(t) = \dfrac{\nu_1 + \nu_2}{2}t + \dfrac{\nu_2 - \nu_1}{4v_{\text{max}}} \log\left(\cosh\left(2v_{\text{max}}\left[t - \dfrac{T}{2}\right]\right)\right) - \dfrac{\nu_2 - \nu_1}{4v_{\text{max}}} \log\left(\cosh\left(v_{\text{max}}T\right)\right)$$

$$ \Delta \phi = \begin{cases} \delta \phi & |\delta \phi| < \pi \\ \delta \phi - \text{sgn}(\delta \phi) \cdot 2\pi & |\delta \phi| \geq \pi \end{cases} $$

$$ \delta \phi = \phi_2 - \phi_1\ \text{mod}\ 2\pi $$

**Conditions**
* $v_{\text{max}}$ must be positive.
* $v_{\text{max}} T \gg 1$ must be true so  that the conditions on transition waveforms outlined in the previous section are _approximately_ correct. In particular, when $v_{\text{max}} T < \text{atanh}(0.98)$, there will be a jump in the amplitude/frequency by more than 1% of the total transition range (i.e. the atom will make a jump larger than 1% of the interatomic distance).

**Comments**
None

## Erf Transition Waveforms

**Parameters**
$$ v_{\text{max}}\text{: Maximum velocity of the atom/Interatomic Distance} $$

**Waveform Description**

$$ \alpha(t) = \dfrac{\alpha_1 + \alpha_2}{2} + \dfrac{\alpha_2 - \alpha_1}{2} \text{erf}\left(\sqrt{\pi}v_{\text{max}} \left[t - \dfrac{T}{2}\right]\right) $$

$$ \nu(t) = \dfrac{\nu_1 + \nu_2}{2} + \dfrac{\nu_2 - \nu_1}{2} \text{erf}\left(\sqrt{\pi}v_{\text{max}} \left[t - \dfrac{T}{2}\right]\right) $$

**Equations**

$$ f(t) = \alpha(t) \sin\left(\phi_1 + 2\pi \tilde{\phi}(t) + \dfrac{\Delta \phi \cdot t}{T}\right) $$

Where:

$$ \alpha(t) = \dfrac{\alpha_1 + \alpha_2}{2} + \dfrac{\alpha_2 - \alpha_1}{2} \text{erf}\left(\sqrt{\pi}v_{\text{max}}\left[t-\dfrac{T}{2}\right]\right) $$

$$ \tilde{\phi}(t) = \dfrac{\nu_1 + \nu_2}{2}t + \dfrac{\nu_2 - \nu_1}{2\sqrt{\pi} v_{\text{max}}}\left[\sqrt{\pi}v_{\text{max}} \left(\bar{t} - \dfrac{T}{2}\right) \text{erf}\left(\sqrt{\pi} v_{\text{max}} \left(\bar{t} - \dfrac{T}{2}\right)\right) + \dfrac{\exp\left(-\left\{\sqrt{\pi}v_{\text{max}}\left(\bar{t}-\dfrac{T}{2}\right)\right\}^2\right)}{\sqrt{\pi}}\right]^{\bar{t} = t}_{\bar{t} = 0} $$

$$ \Delta \phi = \begin{cases} \delta \phi & |\delta \phi| < \pi \\ \delta \phi - \text{sgn}(\delta \phi) \cdot 2\pi & |\delta \phi| \geq \pi \end{cases} $$

$$ \delta \phi = \phi_2 - \phi_1 \ \text{mod}\ 2\pi $$

**Conditions**
* $v_{\text{max}}$ must be positive.
* $v_{\text{max}} T \gg 1$ must be true so  that the conditions on transition waveforms outlined in the previous section are _approximately_ correct. In particular, when $v_{\text{max}} T < 3.29$, there will be a jump in the amplitude/frequency by more than 1% of the total transition range (i.e. the atom will make a jump larger than 1% of the interatomic distance).

**Comments**
None

## Spline Transition Waveform

**Parameters**
None

**Waveform Description**

$$ \alpha(t) = a_1t^3 + b_1t^2 + c_1t + d_1 $$

$$ \nu(t) = a_2t^3 + b_2t^2 + c_2t + d_2 $$

Where the coefficients are chosen such that the condiions of a transition waveform are satisfied exactly.

**Equations**

$$ f(t) = \alpha(t) \sin\left(\phi_1 + 2\pi \tilde{\phi}(t) + \dfrac{\Delta \phi \cdot t}{T}\right) $$

Where:

$$ \alpha(t) = -2(\alpha_2 - \alpha_1) \left(\dfrac{t}{T}\right)^3 + 3(\alpha_2-\alpha_1)\left(\dfrac{t}{T}\right)^2 + \alpha_1 $$

$$ \tilde{\phi}(t) = \dfrac{-(\nu_2 - \nu_1)}{2T^3} t^4 + \dfrac{\nu_2 - \nu_1}{T^2} t^3 + \nu_1 t $$

$$ \Delta \phi = \begin{cases} \delta \phi & |\delta \phi| < \pi \\ \delta \phi - \text{sgn}(\delta \phi) \cdot 2\pi & |\delta \phi| \geq \pi \end{cases} $$

$$ \delta \phi = \phi_2 - \phi_1 - 2\pi \tilde{\phi}(T)\ \text{mod}\ 2\pi $$

**Conditions**
None

**Comments**
The maximum velocity ratio to the interatomic distance of the atom, i.e. $v_{\text{max}}$, achieved by the atom occurs at $t = \dfrac{T}{2}$ where:

$$ v_{\text{max}} = \dfrac{3}{2T(\nu_2 - \nu_1)} $$