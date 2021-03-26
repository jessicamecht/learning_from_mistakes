import higher
import torch
import torch.nn.F as F
def meta_train_full_lbi(model_tgt, model_src, x_tgt, y_tgt, x_src, y_src,
                        x_tgt_val, y_tgt_val, optimizer_tgt, optimizer_src,
                        device):
    model_tgt.train()
    model_src.train()
    eps_A = None
    eps_B = None
    with higher.innerloop_ctx(model_src,
                              optimizer_src) as (fmodel_src,
                                                 foptimizer_src):
        #V*
        yhat_src = fmodel_src(x_src)
        loss_src = F.cross_entropy(yhat_src, y_src, reduction='none')
        print(f"loss_src without weight={loss_src.mean().item():.2f}")
        if eps_A is None:
            eps_A = torch.zeros(loss_src.size(),
                                requires_grad=True).to(device)
        else:
            eps_A.requires_grad = True
        loss_src = torch.mean(eps_A * loss_src)
        print(f'loss_src weight={loss_src.item():.2f}')
        foptimizer_src.step(loss_src)
        with higher.innerloop_ctx(model_tgt,
                                  optimizer_tgt) as (fmodel_tgt,
                                                     foptimizer_tgt):
            #W*
            yhat_tgt = fmodel_tgt(x_tgt)

            loss_tgt = F.cross_entropy(yhat_tgt, y_tgt, reduction='none')
            yhat_src2 = fmodel_tgt(x_src)
            loss_src_2 = F.cross_entropy(yhat_src2,
                                         y_src,
                                         reduction='none')
            if eps_B is None:
                eps_B = torch.zeros(loss_src_2.size(),
                                    requires_grad=True).to(device)
            else:
                eps_B.requires_grad = True
            loss_src_2 = eps_B * loss_src_2 * args.gamma
            final_loss = torch.cat((loss_tgt, loss_src_2), dim=0)
            final_loss = torch.mean(final_loss)

            norm_sum = 0
            for sw, tw in zip(fmodel_src.parameters(),
                              fmodel_tgt.parameters()):
                w_diff = tw - sw
                w_diff_norm = torch.norm(w_diff)
                norm_sum = norm_sum + w_diff_norm**2
            norm_sum = norm_sum * args.lam
            final_loss = final_loss + norm_sum
            foptimizer_tgt.step(final_loss)


            #validation loss happens here
            yhat_tgt_val = fmodel_tgt(x_tgt_val)
            loss_tgt_val = F.cross_entropy(yhat_tgt_val, y_tgt_val)
            '''gradient with respect to A and B'''
            eps_A_grads, eps_B_grads = torch.autograd.grad(
                loss_tgt_val, [eps_A, eps_B])
            eps_A_grads, eps_B_grads = eps_A_grads.detach(
            ), eps_B_grads.detach()
        return eps_A_grads, eps_B_grads